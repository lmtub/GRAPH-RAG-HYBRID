static FlatRange *address_space_lookup(AddressSpace *as, AddrRange addr)

{

    return bsearch(&addr, as->current_map.ranges, as->current_map.nr,

                   sizeof(FlatRange), cmp_flatrange_addr);

}
