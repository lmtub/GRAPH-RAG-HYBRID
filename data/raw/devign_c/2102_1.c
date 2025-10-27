static void process_incoming_migration_bh(void *opaque)
{
    Error *local_err = NULL;
    MigrationIncomingState *mis = opaque;
    /* Make sure all file formats flush their mutable metadata */
    bdrv_invalidate_cache_all(&local_err);
        migrate_set_state(&mis->state, MIGRATION_STATUS_ACTIVE,
                          MIGRATION_STATUS_FAILED);
        error_report_err(local_err);
        migrate_decompress_threads_join();
        exit(EXIT_FAILURE);
    /*
     * This must happen after all error conditions are dealt with and
     * we're sure the VM is going to be running on this host.
     */
    qemu_announce_self();
    /* If global state section was not received or we are in running
       state, we need to obey autostart. Any other state is set with
       runstate_set. */
    if (!global_state_received() ||
        global_state_get_runstate() == RUN_STATE_RUNNING) {
        if (autostart) {
            vm_start();
        } else {
            runstate_set(RUN_STATE_PAUSED);
    } else {
        runstate_set(global_state_get_runstate());
    migrate_decompress_threads_join();
    /*
     * This must happen after any state changes since as soon as an external
     * observer sees this event they might start to prod at the VM assuming
     * it's ready to use.
     */
    migrate_set_state(&mis->state, MIGRATION_STATUS_ACTIVE,
                      MIGRATION_STATUS_COMPLETED);
    qemu_bh_delete(mis->bh);
    migration_incoming_state_destroy();