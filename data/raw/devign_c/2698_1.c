static int postcopy_start(MigrationState *ms, bool *old_vm_running)

{

    int ret;

    QIOChannelBuffer *bioc;

    QEMUFile *fb;

    int64_t time_at_stop = qemu_clock_get_ms(QEMU_CLOCK_REALTIME);

    bool restart_block = false;

    int cur_state = MIGRATION_STATUS_ACTIVE;

    if (!migrate_pause_before_switchover()) {

        migrate_set_state(&ms->state, MIGRATION_STATUS_ACTIVE,

                          MIGRATION_STATUS_POSTCOPY_ACTIVE);

    }



    trace_postcopy_start();

    qemu_mutex_lock_iothread();

    trace_postcopy_start_set_run();



    qemu_system_wakeup_request(QEMU_WAKEUP_REASON_OTHER);

    *old_vm_running = runstate_is_running();

    global_state_store();

    ret = vm_stop_force_state(RUN_STATE_FINISH_MIGRATE);

    if (ret < 0) {

        goto fail;

    }



    ret = migration_maybe_pause(ms, &cur_state,

                                MIGRATION_STATUS_POSTCOPY_ACTIVE);

    if (ret < 0) {

        goto fail;

    }



    ret = bdrv_inactivate_all();

    if (ret < 0) {

        goto fail;

    }

    restart_block = true;



    /*

     * Cause any non-postcopiable, but iterative devices to

     * send out their final data.

     */

    qemu_savevm_state_complete_precopy(ms->to_dst_file, true, false);



    /*

     * in Finish migrate and with the io-lock held everything should

     * be quiet, but we've potentially still got dirty pages and we

     * need to tell the destination to throw any pages it's already received

     * that are dirty

     */

    if (migrate_postcopy_ram()) {

        if (ram_postcopy_send_discard_bitmap(ms)) {

            error_report("postcopy send discard bitmap failed");

            goto fail;

        }

    }



    /*

     * send rest of state - note things that are doing postcopy

     * will notice we're in POSTCOPY_ACTIVE and not actually

     * wrap their state up here

     */

    qemu_file_set_rate_limit(ms->to_dst_file, INT64_MAX);

    if (migrate_postcopy_ram()) {

        /* Ping just for debugging, helps line traces up */

        qemu_savevm_send_ping(ms->to_dst_file, 2);

    }



    /*

     * While loading the device state we may trigger page transfer

     * requests and the fd must be free to process those, and thus

     * the destination must read the whole device state off the fd before

     * it starts processing it.  Unfortunately the ad-hoc migration format

     * doesn't allow the destination to know the size to read without fully

     * parsing it through each devices load-state code (especially the open

     * coded devices that use get/put).

     * So we wrap the device state up in a package with a length at the start;

     * to do this we use a qemu_buf to hold the whole of the device state.

     */

    bioc = qio_channel_buffer_new(4096);

    qio_channel_set_name(QIO_CHANNEL(bioc), "migration-postcopy-buffer");

    fb = qemu_fopen_channel_output(QIO_CHANNEL(bioc));

    object_unref(OBJECT(bioc));



    /*

     * Make sure the receiver can get incoming pages before we send the rest

     * of the state

     */

    qemu_savevm_send_postcopy_listen(fb);



    qemu_savevm_state_complete_precopy(fb, false, false);

    if (migrate_postcopy_ram()) {

        qemu_savevm_send_ping(fb, 3);

    }



    qemu_savevm_send_postcopy_run(fb);



    /* <><> end of stuff going into the package */



    /* Last point of recovery; as soon as we send the package the destination

     * can open devices and potentially start running.

     * Lets just check again we've not got any errors.

     */

    ret = qemu_file_get_error(ms->to_dst_file);

    if (ret) {

        error_report("postcopy_start: Migration stream errored (pre package)");

        goto fail_closefb;

    }



    restart_block = false;



    /* Now send that blob */

    if (qemu_savevm_send_packaged(ms->to_dst_file, bioc->data, bioc->usage)) {

        goto fail_closefb;

    }

    qemu_fclose(fb);



    /* Send a notify to give a chance for anything that needs to happen

     * at the transition to postcopy and after the device state; in particular

     * spice needs to trigger a transition now

     */

    ms->postcopy_after_devices = true;

    notifier_list_notify(&migration_state_notifiers, ms);



    ms->downtime =  qemu_clock_get_ms(QEMU_CLOCK_REALTIME) - time_at_stop;



    qemu_mutex_unlock_iothread();



    if (migrate_postcopy_ram()) {

        /*

         * Although this ping is just for debug, it could potentially be

         * used for getting a better measurement of downtime at the source.

         */

        qemu_savevm_send_ping(ms->to_dst_file, 4);

    }



    if (migrate_release_ram()) {

        ram_postcopy_migrated_memory_release(ms);

    }



    ret = qemu_file_get_error(ms->to_dst_file);

    if (ret) {

        error_report("postcopy_start: Migration stream errored");

        migrate_set_state(&ms->state, MIGRATION_STATUS_POSTCOPY_ACTIVE,

                              MIGRATION_STATUS_FAILED);

    }



    return ret;



fail_closefb:

    qemu_fclose(fb);

fail:

    migrate_set_state(&ms->state, MIGRATION_STATUS_POSTCOPY_ACTIVE,

                          MIGRATION_STATUS_FAILED);

    if (restart_block) {

        /* A failure happened early enough that we know the destination hasn't

         * accessed block devices, so we're safe to recover.

         */

        Error *local_err = NULL;



        bdrv_invalidate_cache_all(&local_err);

        if (local_err) {

            error_report_err(local_err);

        }

    }

    qemu_mutex_unlock_iothread();

    return -1;

}
