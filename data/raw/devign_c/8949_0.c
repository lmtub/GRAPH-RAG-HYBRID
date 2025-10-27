static bool vtd_do_iommu_translate(VTDAddressSpace *vtd_as, PCIBus *bus,

                                   uint8_t devfn, hwaddr addr, bool is_write,

                                   IOMMUTLBEntry *entry)

{

    IntelIOMMUState *s = vtd_as->iommu_state;

    VTDContextEntry ce;

    uint8_t bus_num = pci_bus_num(bus);

    VTDContextCacheEntry *cc_entry = &vtd_as->context_cache_entry;

    uint64_t slpte, page_mask;

    uint32_t level;

    uint16_t source_id = vtd_make_source_id(bus_num, devfn);

    int ret_fr;

    bool is_fpd_set = false;

    bool reads = true;

    bool writes = true;

    uint8_t access_flags;

    VTDIOTLBEntry *iotlb_entry;



    /*

     * We have standalone memory region for interrupt addresses, we

     * should never receive translation requests in this region.

     */

    assert(!vtd_is_interrupt_addr(addr));



    /* Try to fetch slpte form IOTLB */

    iotlb_entry = vtd_lookup_iotlb(s, source_id, addr);

    if (iotlb_entry) {

        trace_vtd_iotlb_page_hit(source_id, addr, iotlb_entry->slpte,

                                 iotlb_entry->domain_id);

        slpte = iotlb_entry->slpte;

        access_flags = iotlb_entry->access_flags;

        page_mask = iotlb_entry->mask;

        goto out;

    }



    /* Try to fetch context-entry from cache first */

    if (cc_entry->context_cache_gen == s->context_cache_gen) {

        trace_vtd_iotlb_cc_hit(bus_num, devfn, cc_entry->context_entry.hi,

                               cc_entry->context_entry.lo,

                               cc_entry->context_cache_gen);

        ce = cc_entry->context_entry;

        is_fpd_set = ce.lo & VTD_CONTEXT_ENTRY_FPD;

    } else {

        ret_fr = vtd_dev_to_context_entry(s, bus_num, devfn, &ce);

        is_fpd_set = ce.lo & VTD_CONTEXT_ENTRY_FPD;

        if (ret_fr) {

            ret_fr = -ret_fr;

            if (is_fpd_set && vtd_is_qualified_fault(ret_fr)) {

                trace_vtd_fault_disabled();

            } else {

                vtd_report_dmar_fault(s, source_id, addr, ret_fr, is_write);

            }

            goto error;

        }

        /* Update context-cache */

        trace_vtd_iotlb_cc_update(bus_num, devfn, ce.hi, ce.lo,

                                  cc_entry->context_cache_gen,

                                  s->context_cache_gen);

        cc_entry->context_entry = ce;

        cc_entry->context_cache_gen = s->context_cache_gen;

    }



    /*

     * We don't need to translate for pass-through context entries.

     * Also, let's ignore IOTLB caching as well for PT devices.

     */

    if (vtd_ce_get_type(&ce) == VTD_CONTEXT_TT_PASS_THROUGH) {

        entry->iova = addr & VTD_PAGE_MASK_4K;

        entry->translated_addr = entry->iova;

        entry->addr_mask = ~VTD_PAGE_MASK_4K;

        entry->perm = IOMMU_RW;

        trace_vtd_translate_pt(source_id, entry->iova);



        /*

         * When this happens, it means firstly caching-mode is not

         * enabled, and this is the first passthrough translation for

         * the device. Let's enable the fast path for passthrough.

         *

         * When passthrough is disabled again for the device, we can

         * capture it via the context entry invalidation, then the

         * IOMMU region can be swapped back.

         */

        vtd_pt_enable_fast_path(s, source_id);



        return true;

    }



    ret_fr = vtd_iova_to_slpte(&ce, addr, is_write, &slpte, &level,

                               &reads, &writes);

    if (ret_fr) {

        ret_fr = -ret_fr;

        if (is_fpd_set && vtd_is_qualified_fault(ret_fr)) {

            trace_vtd_fault_disabled();

        } else {

            vtd_report_dmar_fault(s, source_id, addr, ret_fr, is_write);

        }

        goto error;

    }



    page_mask = vtd_slpt_level_page_mask(level);

    access_flags = IOMMU_ACCESS_FLAG(reads, writes);

    vtd_update_iotlb(s, source_id, VTD_CONTEXT_ENTRY_DID(ce.hi), addr, slpte,

                     access_flags, level);

out:

    entry->iova = addr & page_mask;

    entry->translated_addr = vtd_get_slpte_addr(slpte) & page_mask;

    entry->addr_mask = ~page_mask;

    entry->perm = access_flags;

    return true;



error:

    entry->iova = 0;

    entry->translated_addr = 0;

    entry->addr_mask = 0;

    entry->perm = IOMMU_NONE;

    return false;

}
