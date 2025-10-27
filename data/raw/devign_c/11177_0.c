void memory_region_ram_resize(MemoryRegion *mr, ram_addr_t newsize, Error **errp)

{

    assert(mr->terminates);



    qemu_ram_resize(mr->ram_addr, newsize, errp);

}
