uint8_t *xen_map_cache(target_phys_addr_t phys_addr, target_phys_addr_t size,

                       uint8_t lock)

{

    MapCacheEntry *entry, *pentry = NULL;

    target_phys_addr_t address_index  = phys_addr >> MCACHE_BUCKET_SHIFT;

    target_phys_addr_t address_offset = phys_addr & (MCACHE_BUCKET_SIZE - 1);

    target_phys_addr_t __size = size;



    trace_xen_map_cache(phys_addr);



    if (address_index == mapcache->last_address_index && !lock && !__size) {

        trace_xen_map_cache_return(mapcache->last_address_vaddr + address_offset);

        return mapcache->last_address_vaddr + address_offset;

    }



    /* size is always a multiple of MCACHE_BUCKET_SIZE */

    if ((address_offset + (__size % MCACHE_BUCKET_SIZE)) > MCACHE_BUCKET_SIZE)

        __size += MCACHE_BUCKET_SIZE;

    if (__size % MCACHE_BUCKET_SIZE)

        __size += MCACHE_BUCKET_SIZE - (__size % MCACHE_BUCKET_SIZE);

    if (!__size)

        __size = MCACHE_BUCKET_SIZE;



    entry = &mapcache->entry[address_index % mapcache->nr_buckets];



    while (entry && entry->lock && entry->vaddr_base &&

            (entry->paddr_index != address_index || entry->size != __size ||

             !test_bits(address_offset >> XC_PAGE_SHIFT, size >> XC_PAGE_SHIFT,

                 entry->valid_mapping))) {

        pentry = entry;

        entry = entry->next;

    }

    if (!entry) {

        entry = g_malloc0(sizeof (MapCacheEntry));

        pentry->next = entry;

        xen_remap_bucket(entry, __size, address_index);

    } else if (!entry->lock) {

        if (!entry->vaddr_base || entry->paddr_index != address_index ||

                entry->size != __size ||

                !test_bits(address_offset >> XC_PAGE_SHIFT, size >> XC_PAGE_SHIFT,

                    entry->valid_mapping)) {

            xen_remap_bucket(entry, __size, address_index);

        }

    }



    if(!test_bits(address_offset >> XC_PAGE_SHIFT, size >> XC_PAGE_SHIFT,

                entry->valid_mapping)) {

        mapcache->last_address_index = -1;

        trace_xen_map_cache_return(NULL);

        return NULL;

    }



    mapcache->last_address_index = address_index;

    mapcache->last_address_vaddr = entry->vaddr_base;

    if (lock) {

        MapCacheRev *reventry = g_malloc0(sizeof(MapCacheRev));

        entry->lock++;

        reventry->vaddr_req = mapcache->last_address_vaddr + address_offset;

        reventry->paddr_index = mapcache->last_address_index;

        reventry->size = entry->size;

        QTAILQ_INSERT_HEAD(&mapcache->locked_entries, reventry, next);

    }



    trace_xen_map_cache_return(mapcache->last_address_vaddr + address_offset);

    return mapcache->last_address_vaddr + address_offset;

}
