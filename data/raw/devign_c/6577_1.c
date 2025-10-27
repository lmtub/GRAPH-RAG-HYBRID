CPUArchState *cpu_copy(CPUArchState *env)

{

    CPUState *cpu = ENV_GET_CPU(env);

    CPUState *new_cpu = cpu_init(cpu_model);

    CPUArchState *new_env = cpu->env_ptr;

    CPUBreakpoint *bp;

    CPUWatchpoint *wp;



    /* Reset non arch specific state */

    cpu_reset(new_cpu);



    memcpy(new_env, env, sizeof(CPUArchState));



    /* Clone all break/watchpoints.

       Note: Once we support ptrace with hw-debug register access, make sure

       BP_CPU break/watchpoints are handled correctly on clone. */

    QTAILQ_INIT(&cpu->breakpoints);

    QTAILQ_INIT(&cpu->watchpoints);

    QTAILQ_FOREACH(bp, &cpu->breakpoints, entry) {

        cpu_breakpoint_insert(new_cpu, bp->pc, bp->flags, NULL);

    }

    QTAILQ_FOREACH(wp, &cpu->watchpoints, entry) {

        cpu_watchpoint_insert(new_cpu, wp->vaddr, wp->len, wp->flags, NULL);

    }



    return new_env;

}
