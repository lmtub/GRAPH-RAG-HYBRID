static void pxa2xx_pwrmode_write(CPUARMState *env, const ARMCPRegInfo *ri,

                                 uint64_t value)

{

    PXA2xxState *s = (PXA2xxState *)ri->opaque;

    static const char *pwrmode[8] = {

        "Normal", "Idle", "Deep-idle", "Standby",

        "Sleep", "reserved (!)", "reserved (!)", "Deep-sleep",

    };



    if (value & 8) {

        printf("%s: CPU voltage change attempt\n", __func__);

    }

    switch (value & 7) {

    case 0:

        /* Do nothing */

        break;



    case 1:

        /* Idle */

        if (!(s->cm_regs[CCCR >> 2] & (1U << 31))) { /* CPDIS */

            cpu_interrupt(CPU(s->cpu), CPU_INTERRUPT_HALT);

            break;

        }

        /* Fall through.  */



    case 2:

        /* Deep-Idle */

        cpu_interrupt(CPU(s->cpu), CPU_INTERRUPT_HALT);

        s->pm_regs[RCSR >> 2] |= 0x8; /* Set GPR */

        goto message;



    case 3:

        s->cpu->env.uncached_cpsr = ARM_CPU_MODE_SVC;

        s->cpu->env.daif = PSTATE_A | PSTATE_F | PSTATE_I;

        s->cpu->env.cp15.sctlr_ns = 0;

        s->cpu->env.cp15.c1_coproc = 0;

        s->cpu->env.cp15.ttbr0_el[1] = 0;

        s->cpu->env.cp15.c3 = 0;

        s->pm_regs[PSSR >> 2] |= 0x8; /* Set STS */

        s->pm_regs[RCSR >> 2] |= 0x8; /* Set GPR */



        /*

         * The scratch-pad register is almost universally used

         * for storing the return address on suspend.  For the

         * lack of a resuming bootloader, perform a jump

         * directly to that address.

         */

        memset(s->cpu->env.regs, 0, 4 * 15);

        s->cpu->env.regs[15] = s->pm_regs[PSPR >> 2];



#if 0

        buffer = 0xe59ff000; /* ldr     pc, [pc, #0] */

        cpu_physical_memory_write(0, &buffer, 4);

        buffer = s->pm_regs[PSPR >> 2];

        cpu_physical_memory_write(8, &buffer, 4);

#endif



        /* Suspend */

        cpu_interrupt(current_cpu, CPU_INTERRUPT_HALT);



        goto message;



    default:

    message:

        printf("%s: machine entered %s mode\n", __func__,

               pwrmode[value & 7]);

    }

}
