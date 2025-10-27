int64_t cache_resize(PageCache *cache, int64_t new_num_pages)

{

    PageCache *new_cache;

    int64_t i;



    CacheItem *old_it, *new_it;



    g_assert(cache);



    /* cache was not inited */

    if (cache->page_cache == NULL) {

        return -1;

    }



    /* same size */

    if (pow2floor(new_num_pages) == cache->max_num_items) {

        return cache->max_num_items;

    }



    new_cache = cache_init(new_num_pages, cache->page_size);

    if (!(new_cache)) {

        DPRINTF("Error creating new cache\n");

        return -1;

    }



    /* move all data from old cache */

    for (i = 0; i < cache->max_num_items; i++) {

        old_it = &cache->page_cache[i];

        if (old_it->it_addr != -1) {

            /* check for collision, if there is, keep MRU page */

            new_it = cache_get_by_addr(new_cache, old_it->it_addr);

            if (new_it->it_data) {

                /* keep the MRU page */

                if (new_it->it_age >= old_it->it_age) {

                    g_free(old_it->it_data);

                } else {

                    g_free(new_it->it_data);

                    new_it->it_data = old_it->it_data;

                    new_it->it_age = old_it->it_age;

                    new_it->it_addr = old_it->it_addr;

                }

            } else {

                cache_insert(new_cache, old_it->it_addr, old_it->it_data);

            }

        }

    }




    cache->page_cache = new_cache->page_cache;

    cache->max_num_items = new_cache->max_num_items;

    cache->num_items = new_cache->num_items;



    g_free(new_cache);



    return cache->max_num_items;

}