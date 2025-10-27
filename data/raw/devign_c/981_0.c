void lm32_debug_excp_handler(CPUState *cs)

{

    LM32CPU *cpu = LM32_CPU(cs);

    CPULM32State *env = &cpu->env;

    CPUBreakpoint *bp;



    if (cs->watchpoint_hit) {

        if (cs->watchpoint_hit->flags & BP_CPU) {

            cs->watchpoint_hit = NULL;

            if (check_watchpoints(env)) {

                raise_exception(env, EXCP_WATCHPOINT);

            } else {

                cpu_resume_from_signal(cs, NULL);

            }

        }

    } else {

        QTAILQ_FOREACH(bp, &cs->breakpoints, entry) {

            if (bp->pc == env->pc) {

                if (bp->flags & BP_CPU) {

                    raise_exception(env, EXCP_BREAKPOINT);

                }

                break;

            }

        }

    }

}
