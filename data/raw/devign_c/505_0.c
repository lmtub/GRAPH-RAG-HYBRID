void *address_space_map(AddressSpace *as,

                        target_phys_addr_t addr,

                        target_phys_addr_t *plen,

                        bool is_write)

{

    AddressSpaceDispatch *d = as->dispatch;

    target_phys_addr_t len = *plen;

    target_phys_addr_t todo = 0;

    int l;

    target_phys_addr_t page;

    MemoryRegionSection *section;

    ram_addr_t raddr = RAM_ADDR_MAX;

    ram_addr_t rlen;

    void *ret;



    while (len > 0) {

        page = addr & TARGET_PAGE_MASK;

        l = (page + TARGET_PAGE_SIZE) - addr;

        if (l > len)

            l = len;

        section = phys_page_find(d, page >> TARGET_PAGE_BITS);



        if (!(memory_region_is_ram(section->mr) && !section->readonly)) {

            if (todo || bounce.buffer) {

                break;

            }

            bounce.buffer = qemu_memalign(TARGET_PAGE_SIZE, TARGET_PAGE_SIZE);

            bounce.addr = addr;

            bounce.len = l;

            if (!is_write) {

                address_space_read(as, addr, bounce.buffer, l);

            }



            *plen = l;

            return bounce.buffer;

        }

        if (!todo) {

            raddr = memory_region_get_ram_addr(section->mr)

                + memory_region_section_addr(section, addr);

        }



        len -= l;

        addr += l;

        todo += l;

    }

    rlen = todo;

    ret = qemu_ram_ptr_length(raddr, &rlen);

    *plen = rlen;

    return ret;

}
