static int memory_access_size(MemoryRegion *mr, unsigned l, hwaddr addr)

{

    unsigned access_size_min = mr->ops->impl.min_access_size;

    unsigned access_size_max = mr->ops->impl.max_access_size;



    /* Regions are assumed to support 1-4 byte accesses unless

       otherwise specified.  */

    if (access_size_min == 0) {

        access_size_min = 1;

    }

    if (access_size_max == 0) {

        access_size_max = 4;

    }



    /* Bound the maximum access by the alignment of the address.  */

    if (!mr->ops->impl.unaligned) {

        unsigned align_size_max = addr & -addr;

        if (align_size_max != 0 && align_size_max < access_size_max) {

            access_size_max = align_size_max;

        }

    }



    /* Don't attempt accesses larger than the maximum.  */

    if (l > access_size_max) {

        l = access_size_max;

    }

    /* ??? The users of this function are wrong, not supporting minimums larger

       than the remaining length.  C.f. memory.c:access_with_adjusted_size.  */

    assert(l >= access_size_min);



    return l;

}
