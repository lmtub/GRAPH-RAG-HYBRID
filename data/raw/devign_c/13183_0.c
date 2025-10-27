void qemu_map_cache_init(void)

{

    unsigned long size;

    struct rlimit rlimit_as;



    mapcache = qemu_mallocz(sizeof (MapCache));



    QTAILQ_INIT(&mapcache->locked_entries);

    mapcache->last_address_index = -1;



    getrlimit(RLIMIT_AS, &rlimit_as);

    rlimit_as.rlim_cur = rlimit_as.rlim_max;

    setrlimit(RLIMIT_AS, &rlimit_as);

    mapcache->max_mcache_size = rlimit_as.rlim_max;



    mapcache->nr_buckets =

        (((mapcache->max_mcache_size >> XC_PAGE_SHIFT) +

          (1UL << (MCACHE_BUCKET_SHIFT - XC_PAGE_SHIFT)) - 1) >>

         (MCACHE_BUCKET_SHIFT - XC_PAGE_SHIFT));



    size = mapcache->nr_buckets * sizeof (MapCacheEntry);

    size = (size + XC_PAGE_SIZE - 1) & ~(XC_PAGE_SIZE - 1);

    DPRINTF("qemu_map_cache_init, nr_buckets = %lx size %lu\n", mapcache->nr_buckets, size);

    mapcache->entry = qemu_mallocz(size);

}
