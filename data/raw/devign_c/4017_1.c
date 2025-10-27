static int ram_init_all(RAMState **rsp)

{

    Error *local_err = NULL;



    if (ram_state_init(rsp)) {

        return -1;

    }



    if (migrate_use_xbzrle()) {

        XBZRLE_cache_lock();

        XBZRLE.zero_target_page = g_malloc0(TARGET_PAGE_SIZE);

        XBZRLE.cache = cache_init(migrate_xbzrle_cache_size(),

                                  TARGET_PAGE_SIZE, &local_err);

        if (!XBZRLE.cache) {

            XBZRLE_cache_unlock();

            error_report_err(local_err);

            g_free(*rsp);

            *rsp = NULL;

            return -1;

        }

        XBZRLE_cache_unlock();



        /* We prefer not to abort if there is no memory */

        XBZRLE.encoded_buf = g_try_malloc0(TARGET_PAGE_SIZE);

        if (!XBZRLE.encoded_buf) {

            error_report("Error allocating encoded_buf");

            g_free(*rsp);

            *rsp = NULL;

            return -1;

        }



        XBZRLE.current_buf = g_try_malloc(TARGET_PAGE_SIZE);

        if (!XBZRLE.current_buf) {

            error_report("Error allocating current_buf");

            g_free(XBZRLE.encoded_buf);

            XBZRLE.encoded_buf = NULL;

            g_free(*rsp);

            *rsp = NULL;

            return -1;

        }

    }



    /* For memory_global_dirty_log_start below.  */

    qemu_mutex_lock_iothread();



    qemu_mutex_lock_ramlist();

    rcu_read_lock();



    /* Skip setting bitmap if there is no RAM */

    if (ram_bytes_total()) {

        RAMBlock *block;



        QLIST_FOREACH_RCU(block, &ram_list.blocks, next) {

            unsigned long pages = block->max_length >> TARGET_PAGE_BITS;



            block->bmap = bitmap_new(pages);

            bitmap_set(block->bmap, 0, pages);

            if (migrate_postcopy_ram()) {

                block->unsentmap = bitmap_new(pages);

                bitmap_set(block->unsentmap, 0, pages);

            }

        }

    }



    memory_global_dirty_log_start();

    migration_bitmap_sync(*rsp);

    qemu_mutex_unlock_ramlist();

    qemu_mutex_unlock_iothread();

    rcu_read_unlock();



    return 0;

}
