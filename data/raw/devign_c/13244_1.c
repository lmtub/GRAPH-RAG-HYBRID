static int ram_save_host_page(RAMState *rs, PageSearchStatus *pss,

                              bool last_stage)

{

    int tmppages, pages = 0;

    size_t pagesize_bits =

        qemu_ram_pagesize(pss->block) >> TARGET_PAGE_BITS;



    do {

        tmppages = ram_save_target_page(rs, pss, last_stage);

        if (tmppages < 0) {

            return tmppages;

        }



        pages += tmppages;

        pss->page++;

    } while (pss->page & (pagesize_bits - 1));



    /* The offset we leave with is the last one we looked at */

    pss->page--;

    return pages;

}
