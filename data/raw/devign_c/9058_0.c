PageCache *cache_init(size_t num_pages, size_t page_size)

{

    int64_t i;



    PageCache *cache;



    if (num_pages <= 0) {

        DPRINTF("invalid number of pages\n");

        return NULL;

    }



    /* We prefer not to abort if there is no memory */

    cache = g_try_malloc(sizeof(*cache));

    if (!cache) {

        DPRINTF("Failed to allocate cache\n");

        return NULL;

    }

    /* round down to the nearest power of 2 */

    if (!is_power_of_2(num_pages)) {

        num_pages = pow2floor(num_pages);

        DPRINTF("rounding down to %" PRId64 "\n", num_pages);

    }

    cache->page_size = page_size;

    cache->num_items = 0;

    cache->max_num_items = num_pages;



    DPRINTF("Setting cache buckets to %" PRId64 "\n", cache->max_num_items);



    /* We prefer not to abort if there is no memory */

    cache->page_cache = g_try_malloc((cache->max_num_items) *

                                     sizeof(*cache->page_cache));

    if (!cache->page_cache) {

        DPRINTF("Failed to allocate cache->page_cache\n");

        g_free(cache);

        return NULL;

    }



    for (i = 0; i < cache->max_num_items; i++) {

        cache->page_cache[i].it_data = NULL;

        cache->page_cache[i].it_age = 0;

        cache->page_cache[i].it_addr = -1;

    }



    return cache;

}
