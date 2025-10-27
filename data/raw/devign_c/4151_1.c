static void *migration_thread(void *opaque)

{

    MigrationState *s = opaque;

    int64_t initial_time = qemu_get_clock_ms(rt_clock);

    int64_t initial_bytes = 0;

    int64_t max_size = 0;

    int64_t start_time = initial_time;

    bool old_vm_running = false;



    DPRINTF("beginning savevm\n");

    qemu_savevm_state_begin(s->file, &s->params);



    while (s->state == MIG_STATE_ACTIVE) {

        int64_t current_time;

        uint64_t pending_size;



        if (!qemu_file_rate_limit(s->file)) {

            DPRINTF("iterate\n");

            pending_size = qemu_savevm_state_pending(s->file, max_size);

            DPRINTF("pending size %lu max %lu\n", pending_size, max_size);

            if (pending_size && pending_size >= max_size) {

                qemu_savevm_state_iterate(s->file);

            } else {

                int ret;



                DPRINTF("done iterating\n");

                qemu_mutex_lock_iothread();

                start_time = qemu_get_clock_ms(rt_clock);

                qemu_system_wakeup_request(QEMU_WAKEUP_REASON_OTHER);

                old_vm_running = runstate_is_running();



                ret = vm_stop_force_state(RUN_STATE_FINISH_MIGRATE);

                if (ret >= 0) {

                    qemu_file_set_rate_limit(s->file, INT_MAX);

                    qemu_savevm_state_complete(s->file);

                }

                qemu_mutex_unlock_iothread();



                if (ret < 0) {

                    migrate_finish_set_state(s, MIG_STATE_ERROR);

                    break;

                }



                if (!qemu_file_get_error(s->file)) {

                    migrate_finish_set_state(s, MIG_STATE_COMPLETED);

                    break;

                }

            }

        }



        if (qemu_file_get_error(s->file)) {

            migrate_finish_set_state(s, MIG_STATE_ERROR);

            break;

        }

        current_time = qemu_get_clock_ms(rt_clock);

        if (current_time >= initial_time + BUFFER_DELAY) {

            uint64_t transferred_bytes = qemu_ftell(s->file) - initial_bytes;

            uint64_t time_spent = current_time - initial_time;

            double bandwidth = transferred_bytes / time_spent;

            max_size = bandwidth * migrate_max_downtime() / 1000000;



            s->mbps = time_spent ? (((double) transferred_bytes * 8.0) /

                    ((double) time_spent / 1000.0)) / 1000.0 / 1000.0 : -1;



            DPRINTF("transferred %" PRIu64 " time_spent %" PRIu64

                    " bandwidth %g max_size %" PRId64 "\n",

                    transferred_bytes, time_spent, bandwidth, max_size);

            /* if we haven't sent anything, we don't want to recalculate

               10000 is a small enough number for our purposes */

            if (s->dirty_bytes_rate && transferred_bytes > 10000) {

                s->expected_downtime = s->dirty_bytes_rate / bandwidth;

            }



            qemu_file_reset_rate_limit(s->file);

            initial_time = current_time;

            initial_bytes = qemu_ftell(s->file);

        }

        if (qemu_file_rate_limit(s->file)) {

            /* usleep expects microseconds */

            g_usleep((initial_time + BUFFER_DELAY - current_time)*1000);

        }

    }



    qemu_mutex_lock_iothread();

    if (s->state == MIG_STATE_COMPLETED) {

        int64_t end_time = qemu_get_clock_ms(rt_clock);

        s->total_time = end_time - s->total_time;

        s->downtime = end_time - start_time;

        runstate_set(RUN_STATE_POSTMIGRATE);

    } else {

        if (old_vm_running) {

            vm_start();

        }

    }

    qemu_bh_schedule(s->cleanup_bh);

    qemu_mutex_unlock_iothread();



    return NULL;

}
