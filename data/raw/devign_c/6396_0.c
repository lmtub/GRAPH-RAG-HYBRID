static MemoryRegionSection phys_page_find(target_phys_addr_t index)

{

    uint16_t *p = phys_page_find_alloc(index, 0);

    uint16_t s_index = phys_section_unassigned;

    MemoryRegionSection section;

    target_phys_addr_t delta;



    if (p) {

        s_index = *p;

    }

    section = phys_sections[s_index];

    index <<= TARGET_PAGE_BITS;

    assert(section.offset_within_address_space <= index

           && index <= section.offset_within_address_space + section.size-1);

    delta = index - section.offset_within_address_space;

    section.offset_within_address_space += delta;

    section.offset_within_region += delta;

    section.size -= delta;

    return section;

}
