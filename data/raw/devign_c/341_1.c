void address_space_init(AddressSpace *as, MemoryRegion *root)

{

    memory_region_transaction_begin();

    as->root = root;

    as->current_map = g_new(FlatView, 1);

    flatview_init(as->current_map);



    QTAILQ_INSERT_TAIL(&address_spaces, as, address_spaces_link);

    as->name = NULL;

    memory_region_transaction_commit();

    address_space_init_dispatch(as);

}