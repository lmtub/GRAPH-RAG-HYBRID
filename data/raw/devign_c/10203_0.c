static void sigp_stop_and_store_status(CPUState *cs, run_on_cpu_data arg)

{

    S390CPU *cpu = S390_CPU(cs);

    SigpInfo *si = arg.host_ptr;

    struct kvm_s390_irq irq = {

        .type = KVM_S390_SIGP_STOP,

    };



    /* disabled wait - sleeping in user space */

    if (s390_cpu_get_state(cpu) == CPU_STATE_OPERATING && cs->halted) {

        s390_cpu_set_state(CPU_STATE_STOPPED, cpu);

    }



    switch (s390_cpu_get_state(cpu)) {

    case CPU_STATE_OPERATING:

        cpu->env.sigp_order = SIGP_STOP_STORE_STATUS;

        kvm_s390_vcpu_interrupt(cpu, &irq);

        /* store will be performed when handling the stop intercept */

        break;

    case CPU_STATE_STOPPED:

        /* already stopped, just store the status */

        cpu_synchronize_state(cs);

        kvm_s390_store_status(cpu, KVM_S390_STORE_STATUS_DEF_ADDR, true);

        break;

    }

    si->cc = SIGP_CC_ORDER_CODE_ACCEPTED;

}
