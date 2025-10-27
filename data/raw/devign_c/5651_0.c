void s390_virtio_irq(S390CPU *cpu, int config_change, uint64_t token)

{

    if (kvm_enabled()) {

        kvm_s390_virtio_irq(cpu, config_change, token);

    } else {

        cpu_inject_ext(cpu, EXT_VIRTIO, config_change, token);

    }

}
