static inline uint64_t ldq_phys_internal(hwaddr addr,

                                         enum device_endian endian)

{

    uint8_t *ptr;

    uint64_t val;

    MemoryRegionSection *section;



    section = phys_page_find(address_space_memory.dispatch, addr >> TARGET_PAGE_BITS);



    if (!(memory_region_is_ram(section->mr) ||

          memory_region_is_romd(section->mr))) {

        /* I/O case */

        addr = memory_region_section_addr(section, addr);



        /* XXX This is broken when device endian != cpu endian.

               Fix and add "endian" variable check */

#ifdef TARGET_WORDS_BIGENDIAN

        val = io_mem_read(section->mr, addr, 4) << 32;

        val |= io_mem_read(section->mr, addr + 4, 4);

#else

        val = io_mem_read(section->mr, addr, 4);

        val |= io_mem_read(section->mr, addr + 4, 4) << 32;

#endif

    } else {

        /* RAM case */

        ptr = qemu_get_ram_ptr((memory_region_get_ram_addr(section->mr)

                                & TARGET_PAGE_MASK)

                               + memory_region_section_addr(section, addr));

        switch (endian) {

        case DEVICE_LITTLE_ENDIAN:

            val = ldq_le_p(ptr);

            break;

        case DEVICE_BIG_ENDIAN:

            val = ldq_be_p(ptr);

            break;

        default:

            val = ldq_p(ptr);

            break;

        }

    }

    return val;

}
