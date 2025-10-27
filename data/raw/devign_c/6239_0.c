static void sigp_initial_cpu_reset(CPUState *cs, run_on_cpu_data arg)

{

    S390CPU *cpu = S390_CPU(cs);

    S390CPUClass *scc = S390_CPU_GET_CLASS(cpu);

    SigpInfo *si = arg.host_ptr;



    cpu_synchronize_state(cs);

    scc->initial_cpu_reset(cs);

    cpu_synchronize_post_reset(cs);

    si->cc = SIGP_CC_ORDER_CODE_ACCEPTED;

}
