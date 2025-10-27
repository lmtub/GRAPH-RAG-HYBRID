void xen_map_cache_init(void)

{

    unsigned long size;

    struct rlimit rlimit_as;



    mapcache = g_malloc0(sizeof (MapCache));



    QTAILQ_INIT(&mapcache->locked_entries);

    mapcache->last_address_index = -1;



    getrlimit(RLIMIT_AS, &rlimit_as);

    if (rlimit_as.rlim_max < MCACHE_MAX_SIZE) {

        rlimit_as.rlim_cur = rlimit_as.rlim_max;

    } else {

        rlimit_as.rlim_cur = MCACHE_MAX_SIZE;

    }



    setrlimit(RLIMIT_AS, &rlimit_as);

    mapcache->max_mcache_size = rlimit_as.rlim_cur;



    mapcache->nr_buckets =

        (((mapcache->max_mcache_size >> XC_PAGE_SHIFT) +

          (1UL << (MCACHE_BUCKET_SHIFT - XC_PAGE_SHIFT)) - 1) >>

         (MCACHE_BUCKET_SHIFT - XC_PAGE_SHIFT));



    size = mapcache->nr_buckets * sizeof (MapCacheEntry);

    size = (size + XC_PAGE_SIZE - 1) & ~(XC_PAGE_SIZE - 1);

    DPRINTF("%s, nr_buckets = %lx size %lu\n", __func__,

            mapcache->nr_buckets, size);

    mapcache->entry = g_malloc0(size);

}
