static void as_memory_range_add(AddressSpace *as, FlatRange *fr)

{

    ram_addr_t phys_offset, region_offset;



    memory_region_prepare_ram_addr(fr->mr);



    phys_offset = fr->mr->ram_addr;

    region_offset = fr->offset_in_region;

    /* cpu_register_physical_memory_log() wants region_offset for

     * mmio, but prefers offseting phys_offset for RAM.  Humour it.

     */

    if ((phys_offset & ~TARGET_PAGE_MASK) <= IO_MEM_ROM) {

        phys_offset += region_offset;

        region_offset = 0;

    }



    if (!fr->readable) {

        phys_offset &= ~TARGET_PAGE_MASK & ~IO_MEM_ROMD;

    }



    if (fr->readonly) {

        phys_offset |= IO_MEM_ROM;

    }



    cpu_register_physical_memory_log(int128_get64(fr->addr.start),

                                     int128_get64(fr->addr.size),

                                     phys_offset,

                                     region_offset,

                                     fr->dirty_log_mask);

}
