void xen_invalidate_map_cache_entry(uint8_t *buffer)

{

    MapCacheEntry *entry = NULL, *pentry = NULL;

    MapCacheRev *reventry;

    hwaddr paddr_index;

    hwaddr size;

    int found = 0;



    QTAILQ_FOREACH(reventry, &mapcache->locked_entries, next) {

        if (reventry->vaddr_req == buffer) {

            paddr_index = reventry->paddr_index;

            size = reventry->size;

            found = 1;

            break;

        }

    }

    if (!found) {

        DPRINTF("%s, could not find %p\n", __func__, buffer);

        QTAILQ_FOREACH(reventry, &mapcache->locked_entries, next) {

            DPRINTF("   "TARGET_FMT_plx" -> %p is present\n", reventry->paddr_index, reventry->vaddr_req);

        }

        return;

    }

    QTAILQ_REMOVE(&mapcache->locked_entries, reventry, next);

    g_free(reventry);



    if (mapcache->last_entry != NULL &&

        mapcache->last_entry->paddr_index == paddr_index) {

        mapcache->last_entry = NULL;

    }



    entry = &mapcache->entry[paddr_index % mapcache->nr_buckets];

    while (entry && (entry->paddr_index != paddr_index || entry->size != size)) {

        pentry = entry;

        entry = entry->next;

    }

    if (!entry) {

        DPRINTF("Trying to unmap address %p that is not in the mapcache!\n", buffer);

        return;

    }

    entry->lock--;

    if (entry->lock > 0 || pentry == NULL) {

        return;

    }



    pentry->next = entry->next;

    if (munmap(entry->vaddr_base, entry->size) != 0) {

        perror("unmap fails");

        exit(-1);

    }

    g_free(entry->valid_mapping);

    g_free(entry);

}
