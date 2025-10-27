static void sigp_store_adtl_status(CPUState *cs, run_on_cpu_data arg)

{

    S390CPU *cpu = S390_CPU(cs);

    SigpInfo *si = arg.host_ptr;



    if (!s390_has_feat(S390_FEAT_VECTOR)) {

        set_sigp_status(si, SIGP_STAT_INVALID_ORDER);

        return;

    }



    /* cpu has to be stopped */

    if (s390_cpu_get_state(cpu) != CPU_STATE_STOPPED) {

        set_sigp_status(si, SIGP_STAT_INCORRECT_STATE);

        return;

    }



    /* parameter must be aligned to 1024-byte boundary */

    if (si->param & 0x3ff) {

        set_sigp_status(si, SIGP_STAT_INVALID_PARAMETER);

        return;

    }



    cpu_synchronize_state(cs);



    if (kvm_s390_store_adtl_status(cpu, si->param)) {

        set_sigp_status(si, SIGP_STAT_INVALID_PARAMETER);

        return;

    }

    si->cc = SIGP_CC_ORDER_CODE_ACCEPTED;

}
