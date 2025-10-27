void arm_cpu_do_interrupt(CPUState *cs)

{

    ARMCPU *cpu = ARM_CPU(cs);

    CPUARMState *env = &cpu->env;

    uint32_t addr;

    uint32_t mask;

    int new_mode;

    uint32_t offset;

    uint32_t moe;



    assert(!IS_M(env));



    arm_log_exception(cs->exception_index);



    if (arm_is_psci_call(cpu, cs->exception_index)) {

        arm_handle_psci_call(cpu);

        qemu_log_mask(CPU_LOG_INT, "...handled as PSCI call\n");

        return;

    }



    /* If this is a debug exception we must update the DBGDSCR.MOE bits */

    switch (env->exception.syndrome >> ARM_EL_EC_SHIFT) {

    case EC_BREAKPOINT:

    case EC_BREAKPOINT_SAME_EL:

        moe = 1;

        break;

    case EC_WATCHPOINT:

    case EC_WATCHPOINT_SAME_EL:

        moe = 10;

        break;

    case EC_AA32_BKPT:

        moe = 3;

        break;

    case EC_VECTORCATCH:

        moe = 5;

        break;

    default:

        moe = 0;

        break;

    }



    if (moe) {

        env->cp15.mdscr_el1 = deposit64(env->cp15.mdscr_el1, 2, 4, moe);

    }



    /* TODO: Vectored interrupt controller.  */

    switch (cs->exception_index) {

    case EXCP_UDEF:

        new_mode = ARM_CPU_MODE_UND;

        addr = 0x04;

        mask = CPSR_I;

        if (env->thumb)

            offset = 2;

        else

            offset = 4;

        break;

    case EXCP_SWI:

        if (semihosting_enabled) {

            /* Check for semihosting interrupt.  */

            if (env->thumb) {

                mask = arm_lduw_code(env, env->regs[15] - 2, env->bswap_code)

                    & 0xff;

            } else {

                mask = arm_ldl_code(env, env->regs[15] - 4, env->bswap_code)

                    & 0xffffff;

            }

            /* Only intercept calls from privileged modes, to provide some

               semblance of security.  */

            if (((mask == 0x123456 && !env->thumb)

                    || (mask == 0xab && env->thumb))

                  && (env->uncached_cpsr & CPSR_M) != ARM_CPU_MODE_USR) {

                env->regs[0] = do_arm_semihosting(env);

                qemu_log_mask(CPU_LOG_INT, "...handled as semihosting call\n");

                return;

            }

        }

        new_mode = ARM_CPU_MODE_SVC;

        addr = 0x08;

        mask = CPSR_I;

        /* The PC already points to the next instruction.  */

        offset = 0;

        break;

    case EXCP_BKPT:

        /* See if this is a semihosting syscall.  */

        if (env->thumb && semihosting_enabled) {

            mask = arm_lduw_code(env, env->regs[15], env->bswap_code) & 0xff;

            if (mask == 0xab

                  && (env->uncached_cpsr & CPSR_M) != ARM_CPU_MODE_USR) {

                env->regs[15] += 2;

                env->regs[0] = do_arm_semihosting(env);

                qemu_log_mask(CPU_LOG_INT, "...handled as semihosting call\n");

                return;

            }

        }

        env->exception.fsr = 2;

        /* Fall through to prefetch abort.  */

    case EXCP_PREFETCH_ABORT:

        env->cp15.ifsr_el2 = env->exception.fsr;

        env->cp15.far_el[1] = deposit64(env->cp15.far_el[1], 32, 32,

                                        env->exception.vaddress);

        qemu_log_mask(CPU_LOG_INT, "...with IFSR 0x%x IFAR 0x%x\n",

                      env->cp15.ifsr_el2, (uint32_t)env->exception.vaddress);

        new_mode = ARM_CPU_MODE_ABT;

        addr = 0x0c;

        mask = CPSR_A | CPSR_I;

        offset = 4;

        break;

    case EXCP_DATA_ABORT:

        env->cp15.esr_el[1] = env->exception.fsr;

        env->cp15.far_el[1] = deposit64(env->cp15.far_el[1], 0, 32,

                                        env->exception.vaddress);

        qemu_log_mask(CPU_LOG_INT, "...with DFSR 0x%x DFAR 0x%x\n",

                      (uint32_t)env->cp15.esr_el[1],

                      (uint32_t)env->exception.vaddress);

        new_mode = ARM_CPU_MODE_ABT;

        addr = 0x10;

        mask = CPSR_A | CPSR_I;

        offset = 8;

        break;

    case EXCP_IRQ:

        new_mode = ARM_CPU_MODE_IRQ;

        addr = 0x18;

        /* Disable IRQ and imprecise data aborts.  */

        mask = CPSR_A | CPSR_I;

        offset = 4;

        if (env->cp15.scr_el3 & SCR_IRQ) {

            /* IRQ routed to monitor mode */

            new_mode = ARM_CPU_MODE_MON;

            mask |= CPSR_F;

        }

        break;

    case EXCP_FIQ:

        new_mode = ARM_CPU_MODE_FIQ;

        addr = 0x1c;

        /* Disable FIQ, IRQ and imprecise data aborts.  */

        mask = CPSR_A | CPSR_I | CPSR_F;

        if (env->cp15.scr_el3 & SCR_FIQ) {

            /* FIQ routed to monitor mode */

            new_mode = ARM_CPU_MODE_MON;

        }

        offset = 4;

        break;

    case EXCP_SMC:

        new_mode = ARM_CPU_MODE_MON;

        addr = 0x08;

        mask = CPSR_A | CPSR_I | CPSR_F;

        offset = 0;

        break;

    default:

        cpu_abort(cs, "Unhandled exception 0x%x\n", cs->exception_index);

        return; /* Never happens.  Keep compiler happy.  */

    }



    if (new_mode == ARM_CPU_MODE_MON) {

        addr += env->cp15.mvbar;

    } else if (A32_BANKED_CURRENT_REG_GET(env, sctlr) & SCTLR_V) {

        /* High vectors. When enabled, base address cannot be remapped. */

        addr += 0xffff0000;

    } else {

        /* ARM v7 architectures provide a vector base address register to remap

         * the interrupt vector table.

         * This register is only followed in non-monitor mode, and is banked.

         * Note: only bits 31:5 are valid.

         */

        addr += env->cp15.vbar_el[1];

    }



    if ((env->uncached_cpsr & CPSR_M) == ARM_CPU_MODE_MON) {

        env->cp15.scr_el3 &= ~SCR_NS;

    }



    switch_mode (env, new_mode);

    /* For exceptions taken to AArch32 we must clear the SS bit in both

     * PSTATE and in the old-state value we save to SPSR_<mode>, so zero it now.

     */

    env->uncached_cpsr &= ~PSTATE_SS;

    env->spsr = cpsr_read(env);

    /* Clear IT bits.  */

    env->condexec_bits = 0;

    /* Switch to the new mode, and to the correct instruction set.  */

    env->uncached_cpsr = (env->uncached_cpsr & ~CPSR_M) | new_mode;

    env->daif |= mask;

    /* this is a lie, as the was no c1_sys on V4T/V5, but who cares

     * and we should just guard the thumb mode on V4 */

    if (arm_feature(env, ARM_FEATURE_V4T)) {

        env->thumb = (A32_BANKED_CURRENT_REG_GET(env, sctlr) & SCTLR_TE) != 0;

    }

    env->regs[14] = env->regs[15] + offset;

    env->regs[15] = addr;

    cs->interrupt_request |= CPU_INTERRUPT_EXITTB;

}
