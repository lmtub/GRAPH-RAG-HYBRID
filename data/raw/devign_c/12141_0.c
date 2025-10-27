static inline uint32_t lduw_phys_internal(hwaddr addr,

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

        val = io_mem_read(section->mr, addr, 2);

#if defined(TARGET_WORDS_BIGENDIAN)

        if (endian == DEVICE_LITTLE_ENDIAN) {

            val = bswap16(val);

        }

#else

        if (endian == DEVICE_BIG_ENDIAN) {

            val = bswap16(val);

        }

#endif

    } else {

        /* RAM case */

        ptr = qemu_get_ram_ptr((memory_region_get_ram_addr(section->mr)

                                & TARGET_PAGE_MASK)

                               + memory_region_section_addr(section, addr));

        switch (endian) {

        case DEVICE_LITTLE_ENDIAN:

            val = lduw_le_p(ptr);

            break;

        case DEVICE_BIG_ENDIAN:

            val = lduw_be_p(ptr);

            break;

        default:

            val = lduw_p(ptr);

            break;

        }

    }

    return val;

}
