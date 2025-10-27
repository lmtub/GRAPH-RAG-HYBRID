void xen_invalidate_map_cache(void)

{

    unsigned long i;

    MapCacheRev *reventry;



    /* Flush pending AIO before destroying the mapcache */

    bdrv_drain_all();



    QTAILQ_FOREACH(reventry, &mapcache->locked_entries, next) {

        DPRINTF("There should be no locked mappings at this time, "

                "but "TARGET_FMT_plx" -> %p is present\n",

                reventry->paddr_index, reventry->vaddr_req);

    }



    mapcache_lock();



    for (i = 0; i < mapcache->nr_buckets; i++) {

        MapCacheEntry *entry = &mapcache->entry[i];



        if (entry->vaddr_base == NULL) {

            continue;

        }

        if (entry->lock > 0) {

            continue;

        }



        if (munmap(entry->vaddr_base, entry->size) != 0) {

            perror("unmap fails");

            exit(-1);

        }



        entry->paddr_index = 0;

        entry->vaddr_base = NULL;

        entry->size = 0;

        g_free(entry->valid_mapping);

        entry->valid_mapping = NULL;

    }



    mapcache->last_entry = NULL;



    mapcache_unlock();

}
