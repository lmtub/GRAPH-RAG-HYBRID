void cache_fini(PageCache *cache)

{

    int64_t i;



    g_assert(cache);

    g_assert(cache->page_cache);



    for (i = 0; i < cache->max_num_items; i++) {

        g_free(cache->page_cache[i].it_data);

    }



    g_free(cache->page_cache);

    cache->page_cache = NULL;


}