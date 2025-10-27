void cpu_check_irqs(CPUState *env)

{

    uint32_t pil = env->pil_in | (env->softint & ~SOFTINT_TIMER) |

        ((env->softint & SOFTINT_TIMER) << 14);



    if (pil && (env->interrupt_index == 0 ||

                (env->interrupt_index & ~15) == TT_EXTINT)) {

        unsigned int i;



        for (i = 15; i > 0; i--) {

            if (pil & (1 << i)) {

                int old_interrupt = env->interrupt_index;



                env->interrupt_index = TT_EXTINT | i;

                if (old_interrupt != env->interrupt_index) {

                    CPUIRQ_DPRINTF("Set CPU IRQ %d\n", i);

                    cpu_interrupt(env, CPU_INTERRUPT_HARD);

                }

                break;

            }

        }

    } else if (!pil && (env->interrupt_index & ~15) == TT_EXTINT) {

        CPUIRQ_DPRINTF("Reset CPU IRQ %d\n", env->interrupt_index & 15);

        env->interrupt_index = 0;

        cpu_reset_interrupt(env, CPU_INTERRUPT_HARD);

    }

}
