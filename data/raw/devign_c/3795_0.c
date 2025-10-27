int memory_region_get_fd(MemoryRegion *mr)

{

    if (mr->alias) {

        return memory_region_get_fd(mr->alias);

    }



    assert(mr->terminates);



    return qemu_get_ram_fd(mr->ram_addr & TARGET_PAGE_MASK);

}
