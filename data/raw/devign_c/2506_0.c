static void mtree_print_flatview(fprintf_function p, void *f,

                                 AddressSpace *as)

{

    FlatView *view = address_space_get_flatview(as);

    FlatRange *range = &view->ranges[0];

    MemoryRegion *mr;

    int n = view->nr;



    if (n <= 0) {

        p(f, MTREE_INDENT "No rendered FlatView for "

          "address space '%s'\n", as->name);

        flatview_unref(view);

        return;

    }



    while (n--) {

        mr = range->mr;

        p(f, MTREE_INDENT TARGET_FMT_plx "-"

          TARGET_FMT_plx " (prio %d, %s): %s\n",

          int128_get64(range->addr.start),

          int128_get64(range->addr.start) + MR_SIZE(range->addr.size),

          mr->priority,

          memory_region_type(mr),

          memory_region_name(mr));

        range++;

    }



    flatview_unref(view);

}
