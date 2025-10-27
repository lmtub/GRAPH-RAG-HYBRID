void breakpoint_handler(CPUState *cs)

{

    X86CPU *cpu = X86_CPU(cs);

    CPUX86State *env = &cpu->env;

    CPUBreakpoint *bp;



    if (cs->watchpoint_hit) {

        if (cs->watchpoint_hit->flags & BP_CPU) {

            cs->watchpoint_hit = NULL;

            if (check_hw_breakpoints(env, false)) {

                raise_exception(env, EXCP01_DB);

            } else {

                cpu_resume_from_signal(cs, NULL);

            }

        }

    } else {

        QTAILQ_FOREACH(bp, &cs->breakpoints, entry) {

            if (bp->pc == env->eip) {

                if (bp->flags & BP_CPU) {

                    check_hw_breakpoints(env, true);

                    raise_exception(env, EXCP01_DB);

                }

                break;

            }

        }

    }

}
