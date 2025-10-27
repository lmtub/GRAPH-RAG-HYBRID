static void do_address_space_destroy(AddressSpace *as)
{
    MemoryListener *listener;
    address_space_destroy_dispatch(as);
    QTAILQ_FOREACH(listener, &memory_listeners, link) {
        assert(listener->address_space_filter != as);
    }
    flatview_unref(as->current_map);
    g_free(as->name);
    g_free(as->ioeventfds);
}