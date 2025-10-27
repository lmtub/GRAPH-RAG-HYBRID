void address_space_rw(AddressSpace *as, target_phys_addr_t addr, uint8_t *buf,

                      int len, bool is_write)

{

    AddressSpaceDispatch *d = as->dispatch;

    int l;

    uint8_t *ptr;

    uint32_t val;

    target_phys_addr_t page;

    MemoryRegionSection *section;



    while (len > 0) {

        page = addr & TARGET_PAGE_MASK;

        l = (page + TARGET_PAGE_SIZE) - addr;

        if (l > len)

            l = len;

        section = phys_page_find(d, page >> TARGET_PAGE_BITS);



        if (is_write) {

            if (!memory_region_is_ram(section->mr)) {

                target_phys_addr_t addr1;

                addr1 = memory_region_section_addr(section, addr);

                /* XXX: could force cpu_single_env to NULL to avoid

                   potential bugs */

                if (l >= 4 && ((addr1 & 3) == 0)) {

                    /* 32 bit write access */

                    val = ldl_p(buf);

                    io_mem_write(section->mr, addr1, val, 4);

                    l = 4;

                } else if (l >= 2 && ((addr1 & 1) == 0)) {

                    /* 16 bit write access */

                    val = lduw_p(buf);

                    io_mem_write(section->mr, addr1, val, 2);

                    l = 2;

                } else {

                    /* 8 bit write access */

                    val = ldub_p(buf);

                    io_mem_write(section->mr, addr1, val, 1);

                    l = 1;

                }

            } else if (!section->readonly) {

                ram_addr_t addr1;

                addr1 = memory_region_get_ram_addr(section->mr)

                    + memory_region_section_addr(section, addr);

                /* RAM case */

                ptr = qemu_get_ram_ptr(addr1);

                memcpy(ptr, buf, l);

                invalidate_and_set_dirty(addr1, l);

                qemu_put_ram_ptr(ptr);

            }

        } else {

            if (!(memory_region_is_ram(section->mr) ||

                  memory_region_is_romd(section->mr))) {

                target_phys_addr_t addr1;

                /* I/O case */

                addr1 = memory_region_section_addr(section, addr);

                if (l >= 4 && ((addr1 & 3) == 0)) {

                    /* 32 bit read access */

                    val = io_mem_read(section->mr, addr1, 4);

                    stl_p(buf, val);

                    l = 4;

                } else if (l >= 2 && ((addr1 & 1) == 0)) {

                    /* 16 bit read access */

                    val = io_mem_read(section->mr, addr1, 2);

                    stw_p(buf, val);

                    l = 2;

                } else {

                    /* 8 bit read access */

                    val = io_mem_read(section->mr, addr1, 1);

                    stb_p(buf, val);

                    l = 1;

                }

            } else {

                /* RAM case */

                ptr = qemu_get_ram_ptr(section->mr->ram_addr

                                       + memory_region_section_addr(section,

                                                                    addr));

                memcpy(buf, ptr, l);

                qemu_put_ram_ptr(ptr);

            }

        }

        len -= l;

        buf += l;

        addr += l;

    }

}
