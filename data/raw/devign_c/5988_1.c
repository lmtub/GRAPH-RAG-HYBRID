static void ppc6xx_set_irq (void *opaque, int pin, int level)

{

    CPUState *env = opaque;

    int cur_level;



#if defined(PPC_DEBUG_IRQ)

    if (loglevel & CPU_LOG_INT) {

        fprintf(logfile, "%s: env %p pin %d level %d\n", __func__,

                env, pin, level);

    }

#endif

    cur_level = (env->irq_input_state >> pin) & 1;

    /* Don't generate spurious events */

    if ((cur_level == 1 && level == 0) || (cur_level == 0 && level != 0)) {

        switch (pin) {

        case PPC6xx_INPUT_TBEN:

            /* Level sensitive - active high */

#if defined(PPC_DEBUG_IRQ)

            if (loglevel & CPU_LOG_INT) {

                fprintf(logfile, "%s: %s the time base\n",

                        __func__, level ? "start" : "stop");

            }

#endif

            if (level) {

                cpu_ppc_tb_start(env);

            } else {

                cpu_ppc_tb_stop(env);

            }

        case PPC6xx_INPUT_INT:

            /* Level sensitive - active high */

#if defined(PPC_DEBUG_IRQ)

            if (loglevel & CPU_LOG_INT) {

                fprintf(logfile, "%s: set the external IRQ state to %d\n",

                        __func__, level);

            }

#endif

            ppc_set_irq(env, PPC_INTERRUPT_EXT, level);

            break;

        case PPC6xx_INPUT_SMI:

            /* Level sensitive - active high */

#if defined(PPC_DEBUG_IRQ)

            if (loglevel & CPU_LOG_INT) {

                fprintf(logfile, "%s: set the SMI IRQ state to %d\n",

                        __func__, level);

            }

#endif

            ppc_set_irq(env, PPC_INTERRUPT_SMI, level);

            break;

        case PPC6xx_INPUT_MCP:

            /* Negative edge sensitive */

            /* XXX: TODO: actual reaction may depends on HID0 status

             *            603/604/740/750: check HID0[EMCP]

             */

            if (cur_level == 1 && level == 0) {

#if defined(PPC_DEBUG_IRQ)

                if (loglevel & CPU_LOG_INT) {

                    fprintf(logfile, "%s: raise machine check state\n",

                            __func__);

                }

#endif

                ppc_set_irq(env, PPC_INTERRUPT_MCK, 1);

            }

            break;

        case PPC6xx_INPUT_CKSTP_IN:

            /* Level sensitive - active low */

            /* XXX: TODO: relay the signal to CKSTP_OUT pin */

            /* XXX: Note that the only way to restart the CPU is to reset it */

            if (level) {

#if defined(PPC_DEBUG_IRQ)

                if (loglevel & CPU_LOG_INT) {

                    fprintf(logfile, "%s: stop the CPU\n", __func__);

                }

#endif

                env->halted = 1;

            }

            break;

        case PPC6xx_INPUT_HRESET:

            /* Level sensitive - active low */

            if (level) {

#if 0 // XXX: TOFIX

#if defined(PPC_DEBUG_IRQ)

                if (loglevel & CPU_LOG_INT) {

                    fprintf(logfile, "%s: reset the CPU\n", __func__);

                }

#endif

                cpu_reset(env);

#endif

            }

            break;

        case PPC6xx_INPUT_SRESET:

#if defined(PPC_DEBUG_IRQ)

            if (loglevel & CPU_LOG_INT) {

                fprintf(logfile, "%s: set the RESET IRQ state to %d\n",

                        __func__, level);

            }

#endif

            ppc_set_irq(env, PPC_INTERRUPT_RESET, level);

            break;

        default:

            /* Unknown pin - do nothing */

#if defined(PPC_DEBUG_IRQ)

            if (loglevel & CPU_LOG_INT) {

                fprintf(logfile, "%s: unknown IRQ pin %d\n", __func__, pin);

            }

#endif

            return;

        }

        if (level)

            env->irq_input_state |= 1 << pin;

        else

            env->irq_input_state &= ~(1 << pin);

    }

}
