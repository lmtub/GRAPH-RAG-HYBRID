static int vtd_page_walk(VTDContextEntry *ce, uint64_t start, uint64_t end,

                         vtd_page_walk_hook hook_fn, void *private,

                         bool notify_unmap)

{

    dma_addr_t addr = vtd_ce_get_slpt_base(ce);

    uint32_t level = vtd_ce_get_level(ce);



    if (!vtd_iova_range_check(start, ce)) {

        return -VTD_FR_ADDR_BEYOND_MGAW;

    }



    if (!vtd_iova_range_check(end, ce)) {

        /* Fix end so that it reaches the maximum */

        end = vtd_iova_limit(ce);

    }



    return vtd_page_walk_level(addr, start, end, hook_fn, private,

                               level, true, true, notify_unmap);

}
