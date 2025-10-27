static void vtd_do_iommu_translate(IntelIOMMUState *s, uint8_t bus_num,

                                   uint8_t devfn, hwaddr addr, bool is_write,

                                   IOMMUTLBEntry *entry)

{

    VTDContextEntry ce;

    uint64_t slpte;

    uint32_t level;

    uint16_t source_id = vtd_make_source_id(bus_num, devfn);

    int ret_fr;

    bool is_fpd_set = false;

    bool reads = true;

    bool writes = true;



    /* Check if the request is in interrupt address range */

    if (vtd_is_interrupt_addr(addr)) {

        if (is_write) {

            /* FIXME: since we don't know the length of the access here, we

             * treat Non-DWORD length write requests without PASID as

             * interrupt requests, too. Withoud interrupt remapping support,

             * we just use 1:1 mapping.

             */

            VTD_DPRINTF(MMU, "write request to interrupt address "

                        "gpa 0x%"PRIx64, addr);

            entry->iova = addr & VTD_PAGE_MASK_4K;

            entry->translated_addr = addr & VTD_PAGE_MASK_4K;

            entry->addr_mask = ~VTD_PAGE_MASK_4K;

            entry->perm = IOMMU_WO;

            return;

        } else {

            VTD_DPRINTF(GENERAL, "error: read request from interrupt address "

                        "gpa 0x%"PRIx64, addr);

            vtd_report_dmar_fault(s, source_id, addr, VTD_FR_READ, is_write);

            return;

        }

    }



    ret_fr = vtd_dev_to_context_entry(s, bus_num, devfn, &ce);

    is_fpd_set = ce.lo & VTD_CONTEXT_ENTRY_FPD;

    if (ret_fr) {

        ret_fr = -ret_fr;

        if (is_fpd_set && vtd_is_qualified_fault(ret_fr)) {

            VTD_DPRINTF(FLOG, "fault processing is disabled for DMA requests "

                        "through this context-entry (with FPD Set)");

        } else {

            vtd_report_dmar_fault(s, source_id, addr, ret_fr, is_write);

        }

        return;

    }



    ret_fr = vtd_gpa_to_slpte(&ce, addr, is_write, &slpte, &level,

                              &reads, &writes);

    if (ret_fr) {

        ret_fr = -ret_fr;

        if (is_fpd_set && vtd_is_qualified_fault(ret_fr)) {

            VTD_DPRINTF(FLOG, "fault processing is disabled for DMA requests "

                        "through this context-entry (with FPD Set)");

        } else {

            vtd_report_dmar_fault(s, source_id, addr, ret_fr, is_write);

        }

        return;

    }



    entry->iova = addr & VTD_PAGE_MASK_4K;

    entry->translated_addr = vtd_get_slpte_addr(slpte) & VTD_PAGE_MASK_4K;

    entry->addr_mask = ~VTD_PAGE_MASK_4K;

    entry->perm = (writes ? 2 : 0) + (reads ? 1 : 0);

}
