static void vtd_iotlb_page_invalidate(IntelIOMMUState *s, uint16_t domain_id,

                                      hwaddr addr, uint8_t am)

{

    VTDIOTLBPageInvInfo info;



    assert(am <= VTD_MAMV);

    info.domain_id = domain_id;

    info.gfn = addr >> VTD_PAGE_SHIFT_4K;

    info.mask = ~((1 << am) - 1);

    g_hash_table_foreach_remove(s->iotlb, vtd_hash_remove_by_page, &info);

}
