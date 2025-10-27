void cpu_check_irqs(CPUSPARCState *env)

{

    uint32_t pil = env->pil_in |

                  (env->softint & ~(SOFTINT_TIMER | SOFTINT_STIMER));



    /* check if TM or SM in SOFTINT are set

       setting these also causes interrupt 14 */

    if (env->softint & (SOFTINT_TIMER | SOFTINT_STIMER)) {

        pil |= 1 << 14;

    }



    /* The bit corresponding to psrpil is (1<< psrpil), the next bit

       is (2 << psrpil). */

    if (pil < (2 << env->psrpil)){

        if (env->interrupt_request & CPU_INTERRUPT_HARD) {

            CPUIRQ_DPRINTF("Reset CPU IRQ (current interrupt %x)\n",

                           env->interrupt_index);

            env->interrupt_index = 0;

            cpu_reset_interrupt(env, CPU_INTERRUPT_HARD);

        }

        return;

    }



    if (cpu_interrupts_enabled(env)) {



        unsigned int i;



        for (i = 15; i > env->psrpil; i--) {

            if (pil & (1 << i)) {

                int old_interrupt = env->interrupt_index;

                int new_interrupt = TT_EXTINT | i;



                if (env->tl > 0 && cpu_tsptr(env)->tt > new_interrupt) {

                    CPUIRQ_DPRINTF("Not setting CPU IRQ: TL=%d "

                                   "current %x >= pending %x\n",

                                   env->tl, cpu_tsptr(env)->tt, new_interrupt);

                } else if (old_interrupt != new_interrupt) {

                    env->interrupt_index = new_interrupt;

                    CPUIRQ_DPRINTF("Set CPU IRQ %d old=%x new=%x\n", i,

                                   old_interrupt, new_interrupt);

                    cpu_interrupt(env, CPU_INTERRUPT_HARD);

                }

                break;

            }

        }

    } else if (env->interrupt_request & CPU_INTERRUPT_HARD) {

        CPUIRQ_DPRINTF("Interrupts disabled, pil=%08x pil_in=%08x softint=%08x "

                       "current interrupt %x\n",

                       pil, env->pil_in, env->softint, env->interrupt_index);

        env->interrupt_index = 0;

        cpu_reset_interrupt(env, CPU_INTERRUPT_HARD);

    }

}
