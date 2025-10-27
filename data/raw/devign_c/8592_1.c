static AddrRange addrrange_intersection(AddrRange r1, AddrRange r2)

{

    uint64_t start = MAX(r1.start, r2.start);

    /* off-by-one arithmetic to prevent overflow */

    uint64_t end = MIN(addrrange_end(r1) - 1, addrrange_end(r2) - 1);

    return addrrange_make(start, end - start + 1);

}
