void s390x_cpu_debug_excp_handler(CPUState *cs)

{

    S390CPU *cpu = S390_CPU(cs);

    CPUS390XState *env = &cpu->env;

    CPUWatchpoint *wp_hit = cs->watchpoint_hit;



    if (wp_hit && wp_hit->flags & BP_CPU) {

        /* FIXME: When the storage-alteration-space control bit is set,

           the exception should only be triggered if the memory access

           is done using an address space with the storage-alteration-event

           bit set.  We have no way to detect that with the current

           watchpoint code.  */

        cs->watchpoint_hit = NULL;



        env->per_address = env->psw.addr;

        env->per_perc_atmid |= PER_CODE_EVENT_STORE | get_per_atmid(env);

        /* FIXME: We currently no way to detect the address space used

           to trigger the watchpoint.  For now just consider it is the

           current default ASC. This turn to be true except when MVCP

           and MVCS instrutions are not used.  */

        env->per_perc_atmid |= env->psw.mask & (PSW_MASK_ASC) >> 46;



        /* Remove all watchpoints to re-execute the code.  A PER exception

           will be triggered, it will call load_psw which will recompute

           the watchpoints.  */

        cpu_watchpoint_remove_all(cs, BP_CPU);

        cpu_resume_from_signal(cs, NULL);

    }

}
