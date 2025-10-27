void *memory_region_get_ram_ptr(MemoryRegion *mr)

{

    if (mr->alias) {

        return memory_region_get_ram_ptr(mr->alias) + mr->alias_offset;

    }



    assert(mr->terminates);



    return qemu_get_ram_ptr(mr->ram_addr & TARGET_PAGE_MASK);

}
