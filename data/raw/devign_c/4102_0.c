void ppc_hw_interrupt(CPUPPCState *env)

{

    PowerPCCPU *cpu = ppc_env_get_cpu(env);

    int hdice;

#if 0

    CPUState *cs = CPU(cpu);



    qemu_log_mask(CPU_LOG_INT, "%s: %p pending %08x req %08x me %d ee %d\n",

                  __func__, env, env->pending_interrupts,

                  cs->interrupt_request, (int)msr_me, (int)msr_ee);

#endif

    /* External reset */

    if (env->pending_interrupts & (1 << PPC_INTERRUPT_RESET)) {

        env->pending_interrupts &= ~(1 << PPC_INTERRUPT_RESET);

        powerpc_excp(cpu, env->excp_model, POWERPC_EXCP_RESET);

        return;

    }

    /* Machine check exception */

    if (env->pending_interrupts & (1 << PPC_INTERRUPT_MCK)) {

        env->pending_interrupts &= ~(1 << PPC_INTERRUPT_MCK);

        powerpc_excp(cpu, env->excp_model, POWERPC_EXCP_MCHECK);

        return;

    }

#if 0 /* TODO */

    /* External debug exception */

    if (env->pending_interrupts & (1 << PPC_INTERRUPT_DEBUG)) {

        env->pending_interrupts &= ~(1 << PPC_INTERRUPT_DEBUG);

        powerpc_excp(cpu, env->excp_model, POWERPC_EXCP_DEBUG);

        return;

    }

#endif

    if (0) {

        /* XXX: find a suitable condition to enable the hypervisor mode */

        hdice = env->spr[SPR_LPCR] & 1;

    } else {

        hdice = 0;

    }

    if ((msr_ee != 0 || msr_hv == 0 || msr_pr != 0) && hdice != 0) {

        /* Hypervisor decrementer exception */

        if (env->pending_interrupts & (1 << PPC_INTERRUPT_HDECR)) {

            env->pending_interrupts &= ~(1 << PPC_INTERRUPT_HDECR);

            powerpc_excp(cpu, env->excp_model, POWERPC_EXCP_HDECR);

            return;

        }

    }

    if (msr_ce != 0) {

        /* External critical interrupt */

        if (env->pending_interrupts & (1 << PPC_INTERRUPT_CEXT)) {

            /* Taking a critical external interrupt does not clear the external

             * critical interrupt status

             */

#if 0

            env->pending_interrupts &= ~(1 << PPC_INTERRUPT_CEXT);

#endif

            powerpc_excp(cpu, env->excp_model, POWERPC_EXCP_CRITICAL);

            return;

        }

    }

    if (msr_ee != 0) {

        /* Watchdog timer on embedded PowerPC */

        if (env->pending_interrupts & (1 << PPC_INTERRUPT_WDT)) {

            env->pending_interrupts &= ~(1 << PPC_INTERRUPT_WDT);

            powerpc_excp(cpu, env->excp_model, POWERPC_EXCP_WDT);

            return;

        }

        if (env->pending_interrupts & (1 << PPC_INTERRUPT_CDOORBELL)) {

            env->pending_interrupts &= ~(1 << PPC_INTERRUPT_CDOORBELL);

            powerpc_excp(cpu, env->excp_model, POWERPC_EXCP_DOORCI);

            return;

        }

        /* Fixed interval timer on embedded PowerPC */

        if (env->pending_interrupts & (1 << PPC_INTERRUPT_FIT)) {

            env->pending_interrupts &= ~(1 << PPC_INTERRUPT_FIT);

            powerpc_excp(cpu, env->excp_model, POWERPC_EXCP_FIT);

            return;

        }

        /* Programmable interval timer on embedded PowerPC */

        if (env->pending_interrupts & (1 << PPC_INTERRUPT_PIT)) {

            env->pending_interrupts &= ~(1 << PPC_INTERRUPT_PIT);

            powerpc_excp(cpu, env->excp_model, POWERPC_EXCP_PIT);

            return;

        }

        /* Decrementer exception */

        if (env->pending_interrupts & (1 << PPC_INTERRUPT_DECR)) {

            env->pending_interrupts &= ~(1 << PPC_INTERRUPT_DECR);

            powerpc_excp(cpu, env->excp_model, POWERPC_EXCP_DECR);

            return;

        }

        /* External interrupt */

        if (env->pending_interrupts & (1 << PPC_INTERRUPT_EXT)) {

            /* Taking an external interrupt does not clear the external

             * interrupt status

             */

#if 0

            env->pending_interrupts &= ~(1 << PPC_INTERRUPT_EXT);

#endif

            powerpc_excp(cpu, env->excp_model, POWERPC_EXCP_EXTERNAL);

            return;

        }

        if (env->pending_interrupts & (1 << PPC_INTERRUPT_DOORBELL)) {

            env->pending_interrupts &= ~(1 << PPC_INTERRUPT_DOORBELL);

            powerpc_excp(cpu, env->excp_model, POWERPC_EXCP_DOORI);

            return;

        }

        if (env->pending_interrupts & (1 << PPC_INTERRUPT_PERFM)) {

            env->pending_interrupts &= ~(1 << PPC_INTERRUPT_PERFM);

            powerpc_excp(cpu, env->excp_model, POWERPC_EXCP_PERFM);

            return;

        }

        /* Thermal interrupt */

        if (env->pending_interrupts & (1 << PPC_INTERRUPT_THERM)) {

            env->pending_interrupts &= ~(1 << PPC_INTERRUPT_THERM);

            powerpc_excp(cpu, env->excp_model, POWERPC_EXCP_THERM);

            return;

        }

    }

}
