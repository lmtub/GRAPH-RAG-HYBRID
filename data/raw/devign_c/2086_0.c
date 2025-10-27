void kvm_s390_virtio_irq(S390CPU *cpu, int config_change, uint64_t token)

{

    kvm_s390_interrupt_internal(cpu, KVM_S390_INT_VIRTIO, config_change,

                                token, 1);

}
