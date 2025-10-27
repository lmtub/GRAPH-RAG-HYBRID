static void sigp_cpu_restart(void *arg)

{

    CPUState *cs = arg;

    S390CPU *cpu = S390_CPU(cs);

    struct kvm_s390_irq irq = {

        .type = KVM_S390_RESTART,

    };



    kvm_s390_vcpu_interrupt(cpu, &irq);

    s390_cpu_set_state(CPU_STATE_OPERATING, cpu);

}
