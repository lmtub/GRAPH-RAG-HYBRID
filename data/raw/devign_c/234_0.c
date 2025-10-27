CPUState *cpu_copy(CPUState *env)

{

    CPUState *new_env = cpu_init(env->cpu_model_str);

    CPUState *next_cpu = new_env->next_cpu;

    int cpu_index = new_env->cpu_index;

#if defined(TARGET_HAS_ICE)

    CPUBreakpoint *bp;

    CPUWatchpoint *wp;

#endif



    memcpy(new_env, env, sizeof(CPUState));



    /* Preserve chaining and index. */

    new_env->next_cpu = next_cpu;

    new_env->cpu_index = cpu_index;



    /* Clone all break/watchpoints.

       Note: Once we support ptrace with hw-debug register access, make sure

       BP_CPU break/watchpoints are handled correctly on clone. */

    TAILQ_INIT(&env->breakpoints);

    TAILQ_INIT(&env->watchpoints);

#if defined(TARGET_HAS_ICE)

    TAILQ_FOREACH(bp, &env->breakpoints, entry) {

        cpu_breakpoint_insert(new_env, bp->pc, bp->flags, NULL);

    }

    TAILQ_FOREACH(wp, &env->watchpoints, entry) {

        cpu_watchpoint_insert(new_env, wp->vaddr, (~wp->len_mask) + 1,

                              wp->flags, NULL);

    }

#endif



    return new_env;

}
