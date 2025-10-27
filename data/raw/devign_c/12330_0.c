MemoryRegionSection memory_region_find(MemoryRegion *mr,

                                       hwaddr addr, uint64_t size)

{

    MemoryRegionSection ret = { .mr = NULL };

    MemoryRegion *root;

    AddressSpace *as;

    AddrRange range;

    FlatView *view;

    FlatRange *fr;



    addr += mr->addr;

    for (root = mr; root->parent; ) {

        root = root->parent;

        addr += root->addr;

    }



    as = memory_region_to_address_space(root);

    range = addrrange_make(int128_make64(addr), int128_make64(size));



    view = as->current_map;

    fr = flatview_lookup(view, range);

    if (!fr) {

        return ret;

    }



    while (fr > view->ranges && addrrange_intersects(fr[-1].addr, range)) {

        --fr;

    }



    ret.mr = fr->mr;

    ret.address_space = as;

    range = addrrange_intersection(range, fr->addr);

    ret.offset_within_region = fr->offset_in_region;

    ret.offset_within_region += int128_get64(int128_sub(range.start,

                                                        fr->addr.start));

    ret.size = range.size;

    ret.offset_within_address_space = int128_get64(range.start);

    ret.readonly = fr->readonly;

    memory_region_ref(ret.mr);



    return ret;

}
