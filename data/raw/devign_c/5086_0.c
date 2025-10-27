void memory_region_reset_dirty(MemoryRegion *mr, hwaddr addr,

                               hwaddr size, unsigned client)

{

    assert(mr->terminates);

    cpu_physical_memory_test_and_clear_dirty(mr->ram_addr + addr, size,

                                             client);

}
