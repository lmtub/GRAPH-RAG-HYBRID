int kvm_irqchip_update_msi_route(KVMState *s, int virq, MSIMessage msg,

                                 PCIDevice *dev)

{

    struct kvm_irq_routing_entry kroute = {};



    if (kvm_gsi_direct_mapping()) {

        return 0;

    }



    if (!kvm_irqchip_in_kernel()) {

        return -ENOSYS;

    }



    kroute.gsi = virq;

    kroute.type = KVM_IRQ_ROUTING_MSI;

    kroute.flags = 0;

    kroute.u.msi.address_lo = (uint32_t)msg.address;

    kroute.u.msi.address_hi = msg.address >> 32;

    kroute.u.msi.data = le32_to_cpu(msg.data);

    if (kvm_msi_devid_required()) {

        kroute.flags = KVM_MSI_VALID_DEVID;

        kroute.u.msi.devid = pci_requester_id(dev);

    }

    if (kvm_arch_fixup_msi_route(&kroute, msg.address, msg.data, dev)) {

        return -EINVAL;

    }



    trace_kvm_irqchip_update_msi_route(virq);



    return kvm_update_routing_entry(s, &kroute);

}
