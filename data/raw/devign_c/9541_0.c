static void drive_backup_prepare(BlkTransactionState *common, Error **errp)

{

    DriveBackupState *state = DO_UPCAST(DriveBackupState, common, common);

    BlockBackend *blk;

    DriveBackup *backup;

    Error *local_err = NULL;



    assert(common->action->kind == TRANSACTION_ACTION_KIND_DRIVE_BACKUP);

    backup = common->action->drive_backup;



    blk = blk_by_name(backup->device);

    if (!blk) {

        error_set(errp, ERROR_CLASS_DEVICE_NOT_FOUND,

                  "Device '%s' not found", backup->device);

        return;

    }



    /* AioContext is released in .clean() */

    state->aio_context = blk_get_aio_context(blk);

    aio_context_acquire(state->aio_context);



    qmp_drive_backup(backup->device, backup->target,

                     backup->has_format, backup->format,

                     backup->sync,

                     backup->has_mode, backup->mode,

                     backup->has_speed, backup->speed,

                     backup->has_bitmap, backup->bitmap,

                     backup->has_on_source_error, backup->on_source_error,

                     backup->has_on_target_error, backup->on_target_error,

                     &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        return;

    }



    state->bs = blk_bs(blk);

    state->job = state->bs->job;

}
