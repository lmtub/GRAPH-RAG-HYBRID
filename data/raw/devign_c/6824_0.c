static IOMMUTLBEntry amdvi_translate(MemoryRegion *iommu, hwaddr addr,

                                     bool is_write)

{

    AMDVIAddressSpace *as = container_of(iommu, AMDVIAddressSpace, iommu);

    AMDVIState *s = as->iommu_state;

    IOMMUTLBEntry ret = {

        .target_as = &address_space_memory,

        .iova = addr,

        .translated_addr = 0,

        .addr_mask = ~(hwaddr)0,

        .perm = IOMMU_NONE

    };



    if (!s->enabled) {

        /* AMDVI disabled - corresponds to iommu=off not

         * failure to provide any parameter

         */

        ret.iova = addr & AMDVI_PAGE_MASK_4K;

        ret.translated_addr = addr & AMDVI_PAGE_MASK_4K;

        ret.addr_mask = ~AMDVI_PAGE_MASK_4K;

        ret.perm = IOMMU_RW;

        return ret;

    } else if (amdvi_is_interrupt_addr(addr)) {

        ret.iova = addr & AMDVI_PAGE_MASK_4K;

        ret.translated_addr = addr & AMDVI_PAGE_MASK_4K;

        ret.addr_mask = ~AMDVI_PAGE_MASK_4K;

        ret.perm = IOMMU_WO;

        return ret;

    }



    amdvi_do_translate(as, addr, is_write, &ret);

    trace_amdvi_translation_result(as->bus_num, PCI_SLOT(as->devfn),

            PCI_FUNC(as->devfn), addr, ret.translated_addr);

    return ret;

}
