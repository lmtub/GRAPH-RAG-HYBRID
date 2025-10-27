static void guest_phys_blocks_region_add(MemoryListener *listener,

                                         MemoryRegionSection *section)

{

    GuestPhysListener *g;

    uint64_t section_size;

    hwaddr target_start, target_end;

    uint8_t *host_addr;

    GuestPhysBlock *predecessor;



    /* we only care about RAM */

    if (!memory_region_is_ram(section->mr) ||

        memory_region_is_skip_dump(section->mr)) {

        return;

    }



    g            = container_of(listener, GuestPhysListener, listener);

    section_size = int128_get64(section->size);

    target_start = section->offset_within_address_space;

    target_end   = target_start + section_size;

    host_addr    = memory_region_get_ram_ptr(section->mr) +

                   section->offset_within_region;

    predecessor  = NULL;



    /* find continuity in guest physical address space */

    if (!QTAILQ_EMPTY(&g->list->head)) {

        hwaddr predecessor_size;



        predecessor = QTAILQ_LAST(&g->list->head, GuestPhysBlockHead);

        predecessor_size = predecessor->target_end - predecessor->target_start;



        /* the memory API guarantees monotonically increasing traversal */

        g_assert(predecessor->target_end <= target_start);



        /* we want continuity in both guest-physical and host-virtual memory */

        if (predecessor->target_end < target_start ||

            predecessor->host_addr + predecessor_size != host_addr) {

            predecessor = NULL;

        }

    }



    if (predecessor == NULL) {

        /* isolated mapping, allocate it and add it to the list */

        GuestPhysBlock *block = g_malloc0(sizeof *block);



        block->target_start = target_start;

        block->target_end   = target_end;

        block->host_addr    = host_addr;

        block->mr           = section->mr;

        memory_region_ref(section->mr);



        QTAILQ_INSERT_TAIL(&g->list->head, block, next);

        ++g->list->num;

    } else {

        /* expand predecessor until @target_end; predecessor's start doesn't

         * change

         */

        predecessor->target_end = target_end;

    }



#ifdef DEBUG_GUEST_PHYS_REGION_ADD

    fprintf(stderr, "%s: target_start=" TARGET_FMT_plx " target_end="

            TARGET_FMT_plx ": %s (count: %u)\n", __FUNCTION__, target_start,

            target_end, predecessor ? "joined" : "added", g->list->num);

#endif

}
