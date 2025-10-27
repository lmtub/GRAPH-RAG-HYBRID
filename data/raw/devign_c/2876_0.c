VTDAddressSpace *vtd_find_add_as(IntelIOMMUState *s, PCIBus *bus, int devfn)

{

    uintptr_t key = (uintptr_t)bus;

    VTDBus *vtd_bus = g_hash_table_lookup(s->vtd_as_by_busptr, &key);

    VTDAddressSpace *vtd_dev_as;

    char name[128];



    if (!vtd_bus) {

        uintptr_t *new_key = g_malloc(sizeof(*new_key));

        *new_key = (uintptr_t)bus;

        /* No corresponding free() */

        vtd_bus = g_malloc0(sizeof(VTDBus) + sizeof(VTDAddressSpace *) * \

                            X86_IOMMU_PCI_DEVFN_MAX);

        vtd_bus->bus = bus;

        g_hash_table_insert(s->vtd_as_by_busptr, new_key, vtd_bus);

    }



    vtd_dev_as = vtd_bus->dev_as[devfn];



    if (!vtd_dev_as) {

        snprintf(name, sizeof(name), "intel_iommu_devfn_%d", devfn);

        vtd_bus->dev_as[devfn] = vtd_dev_as = g_malloc0(sizeof(VTDAddressSpace));



        vtd_dev_as->bus = bus;

        vtd_dev_as->devfn = (uint8_t)devfn;

        vtd_dev_as->iommu_state = s;

        vtd_dev_as->context_cache_entry.context_cache_gen = 0;

        memory_region_init_iommu(&vtd_dev_as->iommu, OBJECT(s),

                                 &s->iommu_ops, "intel_iommu", UINT64_MAX);

        memory_region_init_io(&vtd_dev_as->iommu_ir, OBJECT(s),

                              &vtd_mem_ir_ops, s, "intel_iommu_ir",

                              VTD_INTERRUPT_ADDR_SIZE);

        memory_region_add_subregion(&vtd_dev_as->iommu, VTD_INTERRUPT_ADDR_FIRST,

                                    &vtd_dev_as->iommu_ir);

        address_space_init(&vtd_dev_as->as,

                           &vtd_dev_as->iommu, name);

    }

    return vtd_dev_as;

}
