static int qemu_savevm_state(QEMUFile *f, Error **errp)

{

    int ret;

    MigrationParams params = {

        .blk = 0,

        .shared = 0

    };

    MigrationState *ms = migrate_init(&params);

    ms->to_dst_file = f;



    if (qemu_savevm_state_blocked(errp)) {

        return -EINVAL;

    }



    qemu_mutex_unlock_iothread();

    qemu_savevm_state_header(f);

    qemu_savevm_state_begin(f, &params);

    qemu_mutex_lock_iothread();



    while (qemu_file_get_error(f) == 0) {

        if (qemu_savevm_state_iterate(f, false) > 0) {

            break;

        }

    }



    ret = qemu_file_get_error(f);

    if (ret == 0) {

        qemu_savevm_state_complete_precopy(f, false);

        ret = qemu_file_get_error(f);

    }

    qemu_savevm_state_cleanup();

    if (ret != 0) {

        error_setg_errno(errp, -ret, "Error while writing VM state");

    }

    return ret;

}
