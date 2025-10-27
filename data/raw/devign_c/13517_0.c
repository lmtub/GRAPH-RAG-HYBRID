void stl_phys_notdirty(hwaddr addr, uint32_t val)

{

    uint8_t *ptr;

    MemoryRegionSection *section;



    section = phys_page_find(address_space_memory.dispatch, addr >> TARGET_PAGE_BITS);



    if (!memory_region_is_ram(section->mr) || section->readonly) {

        addr = memory_region_section_addr(section, addr);

        if (memory_region_is_ram(section->mr)) {

            section = &phys_sections[phys_section_rom];

        }

        io_mem_write(section->mr, addr, val, 4);

    } else {

        unsigned long addr1 = (memory_region_get_ram_addr(section->mr)

                               & TARGET_PAGE_MASK)

            + memory_region_section_addr(section, addr);

        ptr = qemu_get_ram_ptr(addr1);

        stl_p(ptr, val);



        if (unlikely(in_migration)) {

            if (!cpu_physical_memory_is_dirty(addr1)) {

                /* invalidate code */

                tb_invalidate_phys_page_range(addr1, addr1 + 4, 0);

                /* set dirty bit */

                cpu_physical_memory_set_dirty_flags(

                    addr1, (0xff & ~CODE_DIRTY_FLAG));

            }

        }

    }

}
