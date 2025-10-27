static void mem_begin(MemoryListener *listener)

{

    AddressSpaceDispatch *d = container_of(listener, AddressSpaceDispatch, listener);



    d->phys_map.ptr = PHYS_MAP_NODE_NIL;

}
