static void vtd_realize(DeviceState *dev, Error **errp)

{

    PCMachineState *pcms = PC_MACHINE(qdev_get_machine());

    PCIBus *bus = pcms->bus;

    IntelIOMMUState *s = INTEL_IOMMU_DEVICE(dev);

    X86IOMMUState *x86_iommu = X86_IOMMU_DEVICE(dev);



    VTD_DPRINTF(GENERAL, "");

    x86_iommu->type = TYPE_INTEL;



    if (!vtd_decide_config(s, errp)) {

        return;

    }



    QLIST_INIT(&s->notifiers_list);

    memset(s->vtd_as_by_bus_num, 0, sizeof(s->vtd_as_by_bus_num));

    memory_region_init_io(&s->csrmem, OBJECT(s), &vtd_mem_ops, s,

                          "intel_iommu", DMAR_REG_SIZE);

    sysbus_init_mmio(SYS_BUS_DEVICE(s), &s->csrmem);

    /* No corresponding destroy */

    s->iotlb = g_hash_table_new_full(vtd_uint64_hash, vtd_uint64_equal,

                                     g_free, g_free);

    s->vtd_as_by_busptr = g_hash_table_new_full(vtd_uint64_hash, vtd_uint64_equal,

                                              g_free, g_free);

    vtd_init(s);

    sysbus_mmio_map(SYS_BUS_DEVICE(s), 0, Q35_HOST_BRIDGE_IOMMU_ADDR);

    pci_setup_iommu(bus, vtd_host_dma_iommu, dev);

    /* Pseudo address space under root PCI bus. */

    pcms->ioapic_as = vtd_host_dma_iommu(bus, s, Q35_PSEUDO_DEVFN_IOAPIC);

}
