static int ram_save_init_globals(void)

{

    int64_t ram_bitmap_pages; /* Size of bitmap in pages, including gaps */



    dirty_rate_high_cnt = 0;

    bitmap_sync_count = 0;

    migration_bitmap_sync_init();

    qemu_mutex_init(&migration_bitmap_mutex);



    if (migrate_use_xbzrle()) {

        XBZRLE_cache_lock();

        ZERO_TARGET_PAGE = g_malloc0(TARGET_PAGE_SIZE);

        XBZRLE.cache = cache_init(migrate_xbzrle_cache_size() /

                                  TARGET_PAGE_SIZE,

                                  TARGET_PAGE_SIZE);

        if (!XBZRLE.cache) {

            XBZRLE_cache_unlock();

            error_report("Error creating cache");

            return -1;

        }

        XBZRLE_cache_unlock();



        /* We prefer not to abort if there is no memory */

        XBZRLE.encoded_buf = g_try_malloc0(TARGET_PAGE_SIZE);

        if (!XBZRLE.encoded_buf) {

            error_report("Error allocating encoded_buf");

            return -1;

        }



        XBZRLE.current_buf = g_try_malloc(TARGET_PAGE_SIZE);

        if (!XBZRLE.current_buf) {

            error_report("Error allocating current_buf");

            g_free(XBZRLE.encoded_buf);

            XBZRLE.encoded_buf = NULL;

            return -1;

        }



        acct_clear();

    }



    /* For memory_global_dirty_log_start below.  */

    qemu_mutex_lock_iothread();



    qemu_mutex_lock_ramlist();

    rcu_read_lock();

    bytes_transferred = 0;

    reset_ram_globals();



    ram_bitmap_pages = last_ram_offset() >> TARGET_PAGE_BITS;

    migration_bitmap_rcu = g_new0(struct BitmapRcu, 1);

    migration_bitmap_rcu->bmap = bitmap_new(ram_bitmap_pages);

    bitmap_set(migration_bitmap_rcu->bmap, 0, ram_bitmap_pages);



    if (migrate_postcopy_ram()) {

        migration_bitmap_rcu->unsentmap = bitmap_new(ram_bitmap_pages);

        bitmap_set(migration_bitmap_rcu->unsentmap, 0, ram_bitmap_pages);

    }



    /*

     * Count the total number of pages used by ram blocks not including any

     * gaps due to alignment or unplugs.

     */

    migration_dirty_pages = ram_bytes_total() >> TARGET_PAGE_BITS;



    memory_global_dirty_log_start();

    migration_bitmap_sync();

    qemu_mutex_unlock_ramlist();

    qemu_mutex_unlock_iothread();

    rcu_read_unlock();



    return 0;

}
