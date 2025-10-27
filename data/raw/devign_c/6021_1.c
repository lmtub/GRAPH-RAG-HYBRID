void mtree_info(fprintf_function mon_printf, void *f)

{

    MemoryRegionListHead ml_head;

    MemoryRegionList *ml, *ml2;



    QTAILQ_INIT(&ml_head);



    mon_printf(f, "memory\n");

    mtree_print_mr(mon_printf, f, address_space_memory.root, 0, 0, &ml_head);



    /* print aliased regions */

    QTAILQ_FOREACH(ml, &ml_head, queue) {

        if (!ml->printed) {

            mon_printf(f, "%s\n", ml->mr->name);

            mtree_print_mr(mon_printf, f, ml->mr, 0, 0, &ml_head);

        }

    }



    QTAILQ_FOREACH_SAFE(ml, &ml_head, queue, ml2) {

        g_free(ml2);

    }



    if (address_space_io.root &&

        !QTAILQ_EMPTY(&address_space_io.root->subregions)) {

        QTAILQ_INIT(&ml_head);

        mon_printf(f, "I/O\n");

        mtree_print_mr(mon_printf, f, address_space_io.root, 0, 0, &ml_head);

    }

}
