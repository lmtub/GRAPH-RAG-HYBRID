static void migrate_fd_put_ready(void *opaque)

{

    MigrationState *s = opaque;

    int ret;



    if (s->state != MIG_STATE_ACTIVE) {

        DPRINTF("put_ready returning because of non-active state\n");

        return;

    }



    DPRINTF("iterate\n");

    ret = qemu_savevm_state_iterate(s->mon, s->file);

    if (ret < 0) {

        migrate_fd_error(s);

    } else if (ret == 1) {

        int old_vm_running = runstate_is_running();



        DPRINTF("done iterating\n");

        vm_stop_force_state(RUN_STATE_FINISH_MIGRATE);



        if (qemu_savevm_state_complete(s->mon, s->file) < 0) {

            migrate_fd_error(s);

        } else {

            migrate_fd_completed(s);

        }

        if (s->state != MIG_STATE_COMPLETED) {

            if (old_vm_running) {

                vm_start();

            }

        }

    }

}
