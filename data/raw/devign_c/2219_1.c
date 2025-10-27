void memory_global_sync_dirty_bitmap(MemoryRegion *address_space)

{

    AddressSpace *as = memory_region_to_address_space(address_space);

    FlatRange *fr;



    FOR_EACH_FLAT_RANGE(fr, &as->current_map) {

        MEMORY_LISTENER_UPDATE_REGION(fr, as, Forward, log_sync);

    }

}
