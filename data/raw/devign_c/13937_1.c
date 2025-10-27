bool memory_region_is_unassigned(MemoryRegion *mr)

{

    return mr != &io_mem_ram && mr != &io_mem_rom

        && mr != &io_mem_notdirty && !mr->rom_device

        && mr != &io_mem_watch;

}
