static void cpu_handle_debug_exception(CPUState *env)

{

    CPUWatchpoint *wp;



    if (!env->watchpoint_hit)

        TAILQ_FOREACH(wp, &env->watchpoints, entry)

            wp->flags &= ~BP_WATCHPOINT_HIT;



    if (debug_excp_handler)

        debug_excp_handler(env);

}
