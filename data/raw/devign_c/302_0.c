void cpu_physical_memory_write_rom(target_phys_addr_t addr,

                                   const uint8_t *buf, int len)

{

    AddressSpaceDispatch *d = address_space_memory.dispatch;

    int l;

    uint8_t *ptr;

    target_phys_addr_t page;

    MemoryRegionSection *section;



    while (len > 0) {

        page = addr & TARGET_PAGE_MASK;

        l = (page + TARGET_PAGE_SIZE) - addr;

        if (l > len)

            l = len;

        section = phys_page_find(d, page >> TARGET_PAGE_BITS);



        if (!(memory_region_is_ram(section->mr) ||

              memory_region_is_romd(section->mr))) {

            /* do nothing */

        } else {

            unsigned long addr1;

            addr1 = memory_region_get_ram_addr(section->mr)

                + memory_region_section_addr(section, addr);

            /* ROM/RAM case */

            ptr = qemu_get_ram_ptr(addr1);

            memcpy(ptr, buf, l);

            invalidate_and_set_dirty(addr1, l);

            qemu_put_ram_ptr(ptr);

        }

        len -= l;

        buf += l;

        addr += l;

    }

}
