static void memory_region_update_coalesced_range(MemoryRegion *mr)

{

    FlatRange *fr;

    CoalescedMemoryRange *cmr;

    AddrRange tmp;



    FOR_EACH_FLAT_RANGE(fr, &address_space_memory.current_map) {

        if (fr->mr == mr) {

            qemu_unregister_coalesced_mmio(int128_get64(fr->addr.start),

                                           int128_get64(fr->addr.size));

            QTAILQ_FOREACH(cmr, &mr->coalesced, link) {

                tmp = addrrange_shift(cmr->addr,

                                      int128_sub(fr->addr.start,

                                                 int128_make64(fr->offset_in_region)));

                if (!addrrange_intersects(tmp, fr->addr)) {

                    continue;

                }

                tmp = addrrange_intersection(tmp, fr->addr);

                qemu_register_coalesced_mmio(int128_get64(tmp.start),

                                             int128_get64(tmp.size));

            }

        }

    }

}
