ram_addr_t memory_region_get_ram_addr(MemoryRegion *mr)

{

    assert(mr->backend_registered);

    return mr->ram_addr;

}
