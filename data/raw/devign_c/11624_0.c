void address_space_destroy_dispatch(AddressSpace *as)

{

    AddressSpaceDispatch *d = as->dispatch;



    memory_listener_unregister(&d->listener);

    g_free(d);

    as->dispatch = NULL;

}
