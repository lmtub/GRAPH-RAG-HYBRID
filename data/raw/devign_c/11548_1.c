static void process_incoming_migration_co(void *opaque)

{

    QEMUFile *f = opaque;

    MigrationIncomingState *mis = migration_incoming_get_current();

    PostcopyState ps;

    int ret;



    mis->from_src_file = f;

    mis->largest_page_size = qemu_ram_pagesize_largest();

    postcopy_state_set(POSTCOPY_INCOMING_NONE);

    migrate_set_state(&mis->state, MIGRATION_STATUS_NONE,

                      MIGRATION_STATUS_ACTIVE);

    ret = qemu_loadvm_state(f);



    ps = postcopy_state_get();

    trace_process_incoming_migration_co_end(ret, ps);

    if (ps != POSTCOPY_INCOMING_NONE) {

        if (ps == POSTCOPY_INCOMING_ADVISE) {

            /*

             * Where a migration had postcopy enabled (and thus went to advise)

             * but managed to complete within the precopy period, we can use

             * the normal exit.

             */

            postcopy_ram_incoming_cleanup(mis);

        } else if (ret >= 0) {

            /*

             * Postcopy was started, cleanup should happen at the end of the

             * postcopy thread.

             */

            trace_process_incoming_migration_co_postcopy_end_main();

            return;

        }

        /* Else if something went wrong then just fall out of the normal exit */

    }



    /* we get COLO info, and know if we are in COLO mode */

    if (!ret && migration_incoming_enable_colo()) {

        mis->migration_incoming_co = qemu_coroutine_self();

        qemu_thread_create(&mis->colo_incoming_thread, "COLO incoming",

             colo_process_incoming_thread, mis, QEMU_THREAD_JOINABLE);

        mis->have_colo_incoming_thread = true;

        qemu_coroutine_yield();



        /* Wait checkpoint incoming thread exit before free resource */

        qemu_thread_join(&mis->colo_incoming_thread);

    }



    qemu_fclose(f);

    free_xbzrle_decoded_buf();



    if (ret < 0) {

        migrate_set_state(&mis->state, MIGRATION_STATUS_ACTIVE,

                          MIGRATION_STATUS_FAILED);

        error_report("load of migration failed: %s", strerror(-ret));

        migrate_decompress_threads_join();

        exit(EXIT_FAILURE);

    }



    mis->bh = qemu_bh_new(process_incoming_migration_bh, mis);

    qemu_bh_schedule(mis->bh);

}
