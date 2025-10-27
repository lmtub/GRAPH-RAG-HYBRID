static void *buffered_file_thread(void *opaque)

{

    MigrationState *s = opaque;

    int64_t initial_time = qemu_get_clock_ms(rt_clock);

    int64_t sleep_time = 0;

    int64_t max_size = 0;

    bool last_round = false;

    int ret;



    qemu_mutex_lock_iothread();

    DPRINTF("beginning savevm\n");

    ret = qemu_savevm_state_begin(s->file, &s->params);

    qemu_mutex_unlock_iothread();



    while (ret >= 0) {

        int64_t current_time;

        uint64_t pending_size;



        qemu_mutex_lock_iothread();

        if (s->state != MIG_STATE_ACTIVE) {

            DPRINTF("put_ready returning because of non-active state\n");

            qemu_mutex_unlock_iothread();

            break;

        }

        if (s->complete) {

            qemu_mutex_unlock_iothread();

            break;

        }

        if (s->bytes_xfer < s->xfer_limit) {

            DPRINTF("iterate\n");

            pending_size = qemu_savevm_state_pending(s->file, max_size);

            DPRINTF("pending size %lu max %lu\n", pending_size, max_size);

            if (pending_size && pending_size >= max_size) {

                ret = qemu_savevm_state_iterate(s->file);

                if (ret < 0) {

                    qemu_mutex_unlock_iothread();

                    break;

                }

            } else {

                int old_vm_running = runstate_is_running();

                int64_t start_time, end_time;



                DPRINTF("done iterating\n");

                start_time = qemu_get_clock_ms(rt_clock);

                qemu_system_wakeup_request(QEMU_WAKEUP_REASON_OTHER);

                vm_stop_force_state(RUN_STATE_FINISH_MIGRATE);

                ret = qemu_savevm_state_complete(s->file);

                if (ret < 0) {

                    qemu_mutex_unlock_iothread();

                    break;

                } else {

                    migrate_fd_completed(s);

                }

                end_time = qemu_get_clock_ms(rt_clock);

                s->total_time = end_time - s->total_time;

                s->downtime = end_time - start_time;

                if (s->state != MIG_STATE_COMPLETED) {

                    if (old_vm_running) {

                        vm_start();

                    }

                }

                last_round = true;

            }

        }

        qemu_mutex_unlock_iothread();

        current_time = qemu_get_clock_ms(rt_clock);

        if (current_time >= initial_time + BUFFER_DELAY) {

            uint64_t transferred_bytes = s->bytes_xfer;

            uint64_t time_spent = current_time - initial_time - sleep_time;

            double bandwidth = transferred_bytes / time_spent;

            max_size = bandwidth * migrate_max_downtime() / 1000000;



            DPRINTF("transferred %" PRIu64 " time_spent %" PRIu64

                    " bandwidth %g max_size %" PRId64 "\n",

                    transferred_bytes, time_spent, bandwidth, max_size);

            /* if we haven't sent anything, we don't want to recalculate

               10000 is a small enough number for our purposes */

            if (s->dirty_bytes_rate && transferred_bytes > 10000) {

                s->expected_downtime = s->dirty_bytes_rate / bandwidth;

            }



            s->bytes_xfer = 0;

            sleep_time = 0;

            initial_time = current_time;

        }

        if (!last_round && (s->bytes_xfer >= s->xfer_limit)) {

            /* usleep expects microseconds */

            g_usleep((initial_time + BUFFER_DELAY - current_time)*1000);

            sleep_time += qemu_get_clock_ms(rt_clock) - current_time;

        }

        buffered_flush(s);

        ret = qemu_file_get_error(s->file);

    }



    if (ret < 0) {

        migrate_fd_error(s);

    }

    g_free(s->buffer);

    return NULL;

}
