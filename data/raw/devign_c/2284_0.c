static void migration_end(void)

{

    if (migration_bitmap) {

        memory_global_dirty_log_stop();

        g_free(migration_bitmap);

        migration_bitmap = NULL;

    }



    XBZRLE_cache_lock();

    if (XBZRLE.cache) {

        cache_fini(XBZRLE.cache);

        g_free(XBZRLE.encoded_buf);

        g_free(XBZRLE.current_buf);

        XBZRLE.cache = NULL;

        XBZRLE.encoded_buf = NULL;

        XBZRLE.current_buf = NULL;

    }

    XBZRLE_cache_unlock();

}
