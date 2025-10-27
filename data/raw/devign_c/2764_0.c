static void migration_completion(MigrationState *s, int current_active_state,

                                 bool *old_vm_running,

                                 int64_t *start_time)

{

    int ret;



    if (s->state == MIGRATION_STATUS_ACTIVE) {

        qemu_mutex_lock_iothread();

        *start_time = qemu_clock_get_ms(QEMU_CLOCK_REALTIME);

        qemu_system_wakeup_request(QEMU_WAKEUP_REASON_OTHER);

        *old_vm_running = runstate_is_running();

        ret = global_state_store();



        if (!ret) {

            ret = vm_stop_force_state(RUN_STATE_FINISH_MIGRATE);

            if (ret >= 0) {

                ret = bdrv_inactivate_all();

            }

            if (ret >= 0) {

                qemu_file_set_rate_limit(s->to_dst_file, INT64_MAX);

                qemu_savevm_state_complete_precopy(s->to_dst_file, false);

            }

        }

        qemu_mutex_unlock_iothread();



        if (ret < 0) {

            goto fail;

        }

    } else if (s->state == MIGRATION_STATUS_POSTCOPY_ACTIVE) {

        trace_migration_completion_postcopy_end();



        qemu_savevm_state_complete_postcopy(s->to_dst_file);

        trace_migration_completion_postcopy_end_after_complete();

    }



    /*

     * If rp was opened we must clean up the thread before

     * cleaning everything else up (since if there are no failures

     * it will wait for the destination to send it's status in

     * a SHUT command).

     * Postcopy opens rp if enabled (even if it's not avtivated)

     */

    if (migrate_postcopy_ram()) {

        int rp_error;

        trace_migration_completion_postcopy_end_before_rp();

        rp_error = await_return_path_close_on_source(s);

        trace_migration_completion_postcopy_end_after_rp(rp_error);

        if (rp_error) {

            goto fail_invalidate;

        }

    }



    if (qemu_file_get_error(s->to_dst_file)) {

        trace_migration_completion_file_err();

        goto fail_invalidate;

    }



    migrate_set_state(&s->state, current_active_state,

                      MIGRATION_STATUS_COMPLETED);

    return;



fail_invalidate:

    /* If not doing postcopy, vm_start() will be called: let's regain

     * control on images.

     */

    if (s->state == MIGRATION_STATUS_ACTIVE) {

        Error *local_err = NULL;



        bdrv_invalidate_cache_all(&local_err);

        if (local_err) {

            error_report_err(local_err);

        }

    }



fail:

    migrate_set_state(&s->state, current_active_state,

                      MIGRATION_STATUS_FAILED);

}
