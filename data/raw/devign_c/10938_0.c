static void sigp_start(CPUState *cs, run_on_cpu_data arg)

{

    S390CPU *cpu = S390_CPU(cs);

    SigpInfo *si = arg.host_ptr;



    if (s390_cpu_get_state(cpu) != CPU_STATE_STOPPED) {

        si->cc = SIGP_CC_ORDER_CODE_ACCEPTED;

        return;

    }



    s390_cpu_set_state(CPU_STATE_OPERATING, cpu);

    si->cc = SIGP_CC_ORDER_CODE_ACCEPTED;

}
