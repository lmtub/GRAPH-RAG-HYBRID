static int subpage_register (subpage_t *mmio, uint32_t start, uint32_t end,

                             ram_addr_t memory, ram_addr_t region_offset)

{

    int idx, eidx;



    if (start >= TARGET_PAGE_SIZE || end >= TARGET_PAGE_SIZE)

        return -1;

    idx = SUBPAGE_IDX(start);

    eidx = SUBPAGE_IDX(end);

#if defined(DEBUG_SUBPAGE)

    printf("%s: %p start %08x end %08x idx %08x eidx %08x mem %ld\n", __func__,

           mmio, start, end, idx, eidx, memory);

#endif



    memory = (memory >> IO_MEM_SHIFT) & (IO_MEM_NB_ENTRIES - 1);

    for (; idx <= eidx; idx++) {

        mmio->sub_io_index[idx] = memory;

        mmio->region_offset[idx] = region_offset;

    }



    return 0;

}