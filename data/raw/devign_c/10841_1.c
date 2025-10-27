static uint8_t *xen_map_cache_unlocked(hwaddr phys_addr, hwaddr size,

                                       uint8_t lock, bool dma)

{

    MapCacheEntry *entry, *pentry = NULL;

    hwaddr address_index;

    hwaddr address_offset;

    hwaddr cache_size = size;

    hwaddr test_bit_size;

    bool translated G_GNUC_UNUSED = false;

    bool dummy = false;



tryagain:

    address_index  = phys_addr >> MCACHE_BUCKET_SHIFT;

    address_offset = phys_addr & (MCACHE_BUCKET_SIZE - 1);



    trace_xen_map_cache(phys_addr);



    /* test_bit_size is always a multiple of XC_PAGE_SIZE */

    if (size) {

        test_bit_size = size + (phys_addr & (XC_PAGE_SIZE - 1));



        if (test_bit_size % XC_PAGE_SIZE) {

            test_bit_size += XC_PAGE_SIZE - (test_bit_size % XC_PAGE_SIZE);

        }

    } else {

        test_bit_size = XC_PAGE_SIZE;

    }



    if (mapcache->last_entry != NULL &&

        mapcache->last_entry->paddr_index == address_index &&

        !lock && !size &&

        test_bits(address_offset >> XC_PAGE_SHIFT,

                  test_bit_size >> XC_PAGE_SHIFT,

                  mapcache->last_entry->valid_mapping)) {

        trace_xen_map_cache_return(mapcache->last_entry->vaddr_base + address_offset);

        return mapcache->last_entry->vaddr_base + address_offset;

    }



    /* size is always a multiple of MCACHE_BUCKET_SIZE */

    if (size) {

        cache_size = size + address_offset;

        if (cache_size % MCACHE_BUCKET_SIZE) {

            cache_size += MCACHE_BUCKET_SIZE - (cache_size % MCACHE_BUCKET_SIZE);

        }

    } else {

        cache_size = MCACHE_BUCKET_SIZE;

    }



    entry = &mapcache->entry[address_index % mapcache->nr_buckets];



    while (entry && entry->lock && entry->vaddr_base &&

            (entry->paddr_index != address_index || entry->size != cache_size ||

             !test_bits(address_offset >> XC_PAGE_SHIFT,

                 test_bit_size >> XC_PAGE_SHIFT,

                 entry->valid_mapping))) {

        pentry = entry;

        entry = entry->next;

    }

    if (!entry) {

        entry = g_malloc0(sizeof (MapCacheEntry));

        pentry->next = entry;

        xen_remap_bucket(entry, NULL, cache_size, address_index, dummy);

    } else if (!entry->lock) {

        if (!entry->vaddr_base || entry->paddr_index != address_index ||

                entry->size != cache_size ||

                !test_bits(address_offset >> XC_PAGE_SHIFT,

                    test_bit_size >> XC_PAGE_SHIFT,

                    entry->valid_mapping)) {

            xen_remap_bucket(entry, NULL, cache_size, address_index, dummy);

        }

    }



    if(!test_bits(address_offset >> XC_PAGE_SHIFT,

                test_bit_size >> XC_PAGE_SHIFT,

                entry->valid_mapping)) {

        mapcache->last_entry = NULL;

#ifdef XEN_COMPAT_PHYSMAP

        if (!translated && mapcache->phys_offset_to_gaddr) {

            phys_addr = mapcache->phys_offset_to_gaddr(phys_addr, size, mapcache->opaque);

            translated = true;

            goto tryagain;

        }

#endif

        if (!dummy && runstate_check(RUN_STATE_INMIGRATE)) {

            dummy = true;

            goto tryagain;

        }

        trace_xen_map_cache_return(NULL);

        return NULL;

    }



    mapcache->last_entry = entry;

    if (lock) {

        MapCacheRev *reventry = g_malloc0(sizeof(MapCacheRev));

        entry->lock++;

        reventry->dma = dma;

        reventry->vaddr_req = mapcache->last_entry->vaddr_base + address_offset;

        reventry->paddr_index = mapcache->last_entry->paddr_index;

        reventry->size = entry->size;

        QTAILQ_INSERT_HEAD(&mapcache->locked_entries, reventry, next);

    }



    trace_xen_map_cache_return(mapcache->last_entry->vaddr_base + address_offset);

    return mapcache->last_entry->vaddr_base + address_offset;

}
