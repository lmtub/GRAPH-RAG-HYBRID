int kvm_irqchip_add_msi_route(KVMState *s, MSIMessage msg)

{

    struct kvm_irq_routing_entry kroute;

    int virq;



    if (!kvm_gsi_routing_enabled()) {

        return -ENOSYS;

    }



    virq = kvm_irqchip_get_virq(s);

    if (virq < 0) {

        return virq;

    }



    kroute.gsi = virq;

    kroute.type = KVM_IRQ_ROUTING_MSI;

    kroute.flags = 0;

    kroute.u.msi.address_lo = (uint32_t)msg.address;

    kroute.u.msi.address_hi = msg.address >> 32;

    kroute.u.msi.data = le32_to_cpu(msg.data);



    kvm_add_routing_entry(s, &kroute);

    kvm_irqchip_commit_routes(s);



    return virq;

}
