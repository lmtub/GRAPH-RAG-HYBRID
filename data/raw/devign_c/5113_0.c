static IOMMUTLBEntry spapr_tce_translate_iommu(MemoryRegion *iommu, hwaddr addr,

                                               bool is_write)

{

    sPAPRTCETable *tcet = container_of(iommu, sPAPRTCETable, iommu);

    uint64_t tce;

    IOMMUTLBEntry ret = {

        .target_as = &address_space_memory,

        .iova = 0,

        .translated_addr = 0,

        .addr_mask = ~(hwaddr)0,

        .perm = IOMMU_NONE,

    };



    if ((addr >> tcet->page_shift) < tcet->nb_table) {

        /* Check if we are in bound */

        hwaddr page_mask = IOMMU_PAGE_MASK(tcet->page_shift);



        tce = tcet->table[addr >> tcet->page_shift];

        ret.iova = addr & page_mask;

        ret.translated_addr = tce & page_mask;

        ret.addr_mask = ~page_mask;

        ret.perm = spapr_tce_iommu_access_flags(tce);

    }

    trace_spapr_iommu_xlate(tcet->liobn, addr, ret.iova, ret.perm,

                            ret.addr_mask);



    return ret;

}
