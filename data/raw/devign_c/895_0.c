static inline void stl_phys_internal(hwaddr addr, uint32_t val,

                                     enum device_endian endian)

{

    uint8_t *ptr;

    MemoryRegionSection *section;

    hwaddr l = 4;

    hwaddr addr1;



    section = address_space_translate(&address_space_memory, addr, &addr1, &l,

                                      true);

    if (l < 4 || !memory_region_is_ram(section->mr) || section->readonly) {

        if (memory_region_is_ram(section->mr)) {

            section = &phys_sections[phys_section_rom];

        }

#if defined(TARGET_WORDS_BIGENDIAN)

        if (endian == DEVICE_LITTLE_ENDIAN) {

            val = bswap32(val);

        }

#else

        if (endian == DEVICE_BIG_ENDIAN) {

            val = bswap32(val);

        }

#endif

        io_mem_write(section->mr, addr1, val, 4);

    } else {

        /* RAM case */

        addr1 += memory_region_get_ram_addr(section->mr) & TARGET_PAGE_MASK;

        ptr = qemu_get_ram_ptr(addr1);

        switch (endian) {

        case DEVICE_LITTLE_ENDIAN:

            stl_le_p(ptr, val);

            break;

        case DEVICE_BIG_ENDIAN:

            stl_be_p(ptr, val);

            break;

        default:

            stl_p(ptr, val);

            break;

        }

        invalidate_and_set_dirty(addr1, 4);

    }

}
