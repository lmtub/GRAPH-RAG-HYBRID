address_space_translate_internal(AddressSpaceDispatch *d, hwaddr addr, hwaddr *xlat,

                                 hwaddr *plen, bool resolve_subpage)

{

    MemoryRegionSection *section;

    Int128 diff, diff_page;



    section = address_space_lookup_region(d, addr, resolve_subpage);

    /* Compute offset within MemoryRegionSection */

    addr -= section->offset_within_address_space;



    /* Compute offset within MemoryRegion */

    *xlat = addr + section->offset_within_region;



    diff_page = int128_make64(((addr & TARGET_PAGE_MASK) + TARGET_PAGE_SIZE) - addr);

    diff = int128_sub(section->mr->size, int128_make64(addr));

    diff = int128_min(diff, diff_page);

    *plen = int128_get64(int128_min(diff, int128_make64(*plen)));

    return section;

}
