MemoryRegionSection memory_region_find(MemoryRegion *address_space,

                                       hwaddr addr, uint64_t size)

{

    AddressSpace *as = memory_region_to_address_space(address_space);

    AddrRange range = addrrange_make(int128_make64(addr),

                                     int128_make64(size));

    FlatRange *fr = address_space_lookup(as, range);

    MemoryRegionSection ret = { .mr = NULL, .size = 0 };



    if (!fr) {

        return ret;

    }



    while (fr > as->current_map->ranges

           && addrrange_intersects(fr[-1].addr, range)) {

        --fr;

    }



    ret.mr = fr->mr;

    range = addrrange_intersection(range, fr->addr);

    ret.offset_within_region = fr->offset_in_region;

    ret.offset_within_region += int128_get64(int128_sub(range.start,

                                                        fr->addr.start));

    ret.size = int128_get64(range.size);

    ret.offset_within_address_space = int128_get64(range.start);

    ret.readonly = fr->readonly;

    return ret;

}
