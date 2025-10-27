static AddressSpace *q35_host_dma_iommu(PCIBus *bus, void *opaque, int devfn)

{

    IntelIOMMUState *s = opaque;

    VTDAddressSpace **pvtd_as;

    int bus_num = pci_bus_num(bus);



    assert(0 <= bus_num && bus_num <= VTD_PCI_BUS_MAX);

    assert(0 <= devfn && devfn <= VTD_PCI_DEVFN_MAX);



    pvtd_as = s->address_spaces[bus_num];

    if (!pvtd_as) {

        /* No corresponding free() */

        pvtd_as = g_malloc0(sizeof(VTDAddressSpace *) * VTD_PCI_DEVFN_MAX);

        s->address_spaces[bus_num] = pvtd_as;

    }

    if (!pvtd_as[devfn]) {

        pvtd_as[devfn] = g_malloc0(sizeof(VTDAddressSpace));



        pvtd_as[devfn]->bus_num = (uint8_t)bus_num;

        pvtd_as[devfn]->devfn = (uint8_t)devfn;

        pvtd_as[devfn]->iommu_state = s;

        pvtd_as[devfn]->context_cache_entry.context_cache_gen = 0;

        memory_region_init_iommu(&pvtd_as[devfn]->iommu, OBJECT(s),

                                 &s->iommu_ops, "intel_iommu", UINT64_MAX);

        address_space_init(&pvtd_as[devfn]->as,

                           &pvtd_as[devfn]->iommu, "intel_iommu");

    }

    return &pvtd_as[devfn]->as;

}
