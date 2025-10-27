int kvm_s390_vcpu_interrupt_post_load(S390CPU *cpu)

{

    CPUState *cs = CPU(cpu);

    struct kvm_s390_irq_state irq_state;

    int r;



    if (!kvm_check_extension(kvm_state, KVM_CAP_S390_IRQ_STATE)) {

        return -ENOSYS;

    }



    if (cpu->irqstate_saved_size == 0) {

        return 0;

    }

    irq_state.buf = (uint64_t) cpu->irqstate;

    irq_state.len = cpu->irqstate_saved_size;



    r = kvm_vcpu_ioctl(cs, KVM_S390_SET_IRQ_STATE, &irq_state);

    if (r) {

        error_report("Setting interrupt state failed %d", r);

    }

    return r;

}
