static void register_multipage(AddressSpaceDispatch *d,

                               MemoryRegionSection *section)

{

    hwaddr start_addr = section->offset_within_address_space;

    uint16_t section_index = phys_section_add(section);

    uint64_t num_pages = int128_get64(int128_rshift(section->size,

                                                    TARGET_PAGE_BITS));



    assert(num_pages);

    phys_page_set(d, start_addr >> TARGET_PAGE_BITS, num_pages, section_index);

}
