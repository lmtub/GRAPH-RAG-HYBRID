static target_ulong put_tce_emu(sPAPRTCETable *tcet, target_ulong ioba,

                                target_ulong tce)

{

    IOMMUTLBEntry entry;

    hwaddr page_mask = IOMMU_PAGE_MASK(tcet->page_shift);

    unsigned long index = (ioba - tcet->bus_offset) >> tcet->page_shift;



    if (index >= tcet->nb_table) {

        hcall_dprintf("spapr_vio_put_tce on out-of-bounds IOBA 0x"

                      TARGET_FMT_lx "\n", ioba);

        return H_PARAMETER;

    }



    tcet->table[index] = tce;



    entry.target_as = &address_space_memory,

    entry.iova = ioba & page_mask;

    entry.translated_addr = tce & page_mask;

    entry.addr_mask = ~page_mask;

    entry.perm = spapr_tce_iommu_access_flags(tce);

    memory_region_notify_iommu(&tcet->iommu, entry);



    return H_SUCCESS;

}
