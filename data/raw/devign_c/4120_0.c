void arm_v7m_cpu_do_interrupt(CPUState *cs)

{

    ARMCPU *cpu = ARM_CPU(cs);

    CPUARMState *env = &cpu->env;

    uint32_t lr;



    arm_log_exception(cs->exception_index);



    /* For exceptions we just mark as pending on the NVIC, and let that

       handle it.  */

    switch (cs->exception_index) {

    case EXCP_UDEF:

        armv7m_nvic_set_pending(env->nvic, ARMV7M_EXCP_USAGE);

        env->v7m.cfsr[env->v7m.secure] |= R_V7M_CFSR_UNDEFINSTR_MASK;

        break;

    case EXCP_NOCP:

        armv7m_nvic_set_pending(env->nvic, ARMV7M_EXCP_USAGE);

        env->v7m.cfsr[env->v7m.secure] |= R_V7M_CFSR_NOCP_MASK;

        break;

    case EXCP_INVSTATE:

        armv7m_nvic_set_pending(env->nvic, ARMV7M_EXCP_USAGE);

        env->v7m.cfsr[env->v7m.secure] |= R_V7M_CFSR_INVSTATE_MASK;

        break;

    case EXCP_SWI:

        /* The PC already points to the next instruction.  */

        armv7m_nvic_set_pending(env->nvic, ARMV7M_EXCP_SVC);

        break;

    case EXCP_PREFETCH_ABORT:

    case EXCP_DATA_ABORT:

        /* Note that for M profile we don't have a guest facing FSR, but

         * the env->exception.fsr will be populated by the code that

         * raises the fault, in the A profile short-descriptor format.

         */

        switch (env->exception.fsr & 0xf) {

        case 0x8: /* External Abort */

            switch (cs->exception_index) {

            case EXCP_PREFETCH_ABORT:

                env->v7m.cfsr[M_REG_NS] |= R_V7M_CFSR_IBUSERR_MASK;

                qemu_log_mask(CPU_LOG_INT, "...with CFSR.IBUSERR\n");

                break;

            case EXCP_DATA_ABORT:

                env->v7m.cfsr[M_REG_NS] |=

                    (R_V7M_CFSR_PRECISERR_MASK | R_V7M_CFSR_BFARVALID_MASK);

                env->v7m.bfar = env->exception.vaddress;

                qemu_log_mask(CPU_LOG_INT,

                              "...with CFSR.PRECISERR and BFAR 0x%x\n",

                              env->v7m.bfar);

                break;

            }

            armv7m_nvic_set_pending(env->nvic, ARMV7M_EXCP_BUS);

            break;

        default:

            /* All other FSR values are either MPU faults or "can't happen

             * for M profile" cases.

             */

            switch (cs->exception_index) {

            case EXCP_PREFETCH_ABORT:

                env->v7m.cfsr[env->v7m.secure] |= R_V7M_CFSR_IACCVIOL_MASK;

                qemu_log_mask(CPU_LOG_INT, "...with CFSR.IACCVIOL\n");

                break;

            case EXCP_DATA_ABORT:

                env->v7m.cfsr[env->v7m.secure] |=

                    (R_V7M_CFSR_DACCVIOL_MASK | R_V7M_CFSR_MMARVALID_MASK);

                env->v7m.mmfar[env->v7m.secure] = env->exception.vaddress;

                qemu_log_mask(CPU_LOG_INT,

                              "...with CFSR.DACCVIOL and MMFAR 0x%x\n",

                              env->v7m.mmfar[env->v7m.secure]);

                break;

            }

            armv7m_nvic_set_pending(env->nvic, ARMV7M_EXCP_MEM);

            break;

        }

        break;

    case EXCP_BKPT:

        if (semihosting_enabled()) {

            int nr;

            nr = arm_lduw_code(env, env->regs[15], arm_sctlr_b(env)) & 0xff;

            if (nr == 0xab) {

                env->regs[15] += 2;

                qemu_log_mask(CPU_LOG_INT,

                              "...handling as semihosting call 0x%x\n",

                              env->regs[0]);

                env->regs[0] = do_arm_semihosting(env);

                return;

            }

        }

        armv7m_nvic_set_pending(env->nvic, ARMV7M_EXCP_DEBUG);

        break;

    case EXCP_IRQ:

        break;

    case EXCP_EXCEPTION_EXIT:

        do_v7m_exception_exit(cpu);

        return;

    default:

        cpu_abort(cs, "Unhandled exception 0x%x\n", cs->exception_index);

        return; /* Never happens.  Keep compiler happy.  */

    }



    lr = 0xfffffff1;

    if (env->v7m.control[env->v7m.secure] & R_V7M_CONTROL_SPSEL_MASK) {

        lr |= 4;

    }

    if (!arm_v7m_is_handler_mode(env)) {

        lr |= 8;

    }



    v7m_push_stack(cpu);

    v7m_exception_taken(cpu, lr);

    qemu_log_mask(CPU_LOG_INT, "... as %d\n", env->v7m.exception);

}
