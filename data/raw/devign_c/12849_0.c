void address_space_init(AddressSpace *as, MemoryRegion *root, const char *name)

{

    if (QTAILQ_EMPTY(&address_spaces)) {

        memory_init();

    }



    memory_region_transaction_begin();

    as->root = root;

    as->current_map = g_new(FlatView, 1);

    flatview_init(as->current_map);

    as->ioeventfd_nb = 0;

    as->ioeventfds = NULL;

    QTAILQ_INSERT_TAIL(&address_spaces, as, address_spaces_link);

    as->name = g_strdup(name ? name : "anonymous");

    address_space_init_dispatch(as);

    memory_region_update_pending |= root->enabled;

    memory_region_transaction_commit();

}
