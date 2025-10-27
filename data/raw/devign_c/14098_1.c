int kvm_arch_fixup_msi_route(struct kvm_irq_routing_entry *route,

                             uint64_t address, uint32_t data, PCIDevice *dev)

{

    S390PCIBusDevice *pbdev;

    uint32_t fid = data >> ZPCI_MSI_VEC_BITS;

    uint32_t vec = data & ZPCI_MSI_VEC_MASK;



    pbdev = s390_pci_find_dev_by_fid(fid);

    if (!pbdev) {

        DPRINTF("add_msi_route no dev\n");

        return -ENODEV;

    }



    pbdev->routes.adapter.ind_offset = vec;



    route->type = KVM_IRQ_ROUTING_S390_ADAPTER;

    route->flags = 0;

    route->u.adapter.summary_addr = pbdev->routes.adapter.summary_addr;

    route->u.adapter.ind_addr = pbdev->routes.adapter.ind_addr;

    route->u.adapter.summary_offset = pbdev->routes.adapter.summary_offset;

    route->u.adapter.ind_offset = pbdev->routes.adapter.ind_offset;

    route->u.adapter.adapter_id = pbdev->routes.adapter.adapter_id;

    return 0;

}
