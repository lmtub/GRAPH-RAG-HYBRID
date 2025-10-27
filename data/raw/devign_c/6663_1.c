static int ram_save_host_page(RAMState *rs, PageSearchStatus *pss,

                              bool last_stage,

                              ram_addr_t dirty_ram_abs)

{

    int tmppages, pages = 0;

    size_t pagesize = qemu_ram_pagesize(pss->block);



    do {

        tmppages = ram_save_target_page(rs, pss, last_stage, dirty_ram_abs);

        if (tmppages < 0) {

            return tmppages;

        }



        pages += tmppages;

        pss->offset += TARGET_PAGE_SIZE;

        dirty_ram_abs += TARGET_PAGE_SIZE;

    } while (pss->offset & (pagesize - 1));



    /* The offset we leave with is the last one we looked at */

    pss->offset -= TARGET_PAGE_SIZE;

    return pages;

}
