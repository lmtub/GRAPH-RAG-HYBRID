int kvm_irqchip_update_msi_route(KVMState *s, int virq, MSIMessage msg)

{

    struct kvm_irq_routing_entry kroute;



    if (!kvm_irqchip_in_kernel()) {

        return -ENOSYS;

    }



    kroute.gsi = virq;

    kroute.type = KVM_IRQ_ROUTING_MSI;

    kroute.flags = 0;

    kroute.u.msi.address_lo = (uint32_t)msg.address;

    kroute.u.msi.address_hi = msg.address >> 32;

    kroute.u.msi.data = le32_to_cpu(msg.data);



    return kvm_update_routing_entry(s, &kroute);

}
