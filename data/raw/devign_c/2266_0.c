void stl_phys_notdirty(AddressSpace *as, hwaddr addr, uint32_t val)

{

    uint8_t *ptr;

    MemoryRegion *mr;

    hwaddr l = 4;

    hwaddr addr1;



    mr = address_space_translate(as, addr, &addr1, &l,

                                 true);

    if (l < 4 || !memory_access_is_direct(mr, true)) {

        io_mem_write(mr, addr1, val, 4);

    } else {

        addr1 += memory_region_get_ram_addr(mr) & TARGET_PAGE_MASK;

        ptr = qemu_get_ram_ptr(addr1);

        stl_p(ptr, val);



        if (unlikely(in_migration)) {

            if (cpu_physical_memory_is_clean(addr1)) {

                /* invalidate code */

                tb_invalidate_phys_page_range(addr1, addr1 + 4, 0);

                /* set dirty bit */

                cpu_physical_memory_set_dirty_flag(addr1,

                                                   DIRTY_MEMORY_MIGRATION);

                cpu_physical_memory_set_dirty_flag(addr1, DIRTY_MEMORY_VGA);

            }

        }

    }

}
