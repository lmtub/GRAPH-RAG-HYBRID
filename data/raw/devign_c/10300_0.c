static void blockdev_backup_prepare(BlkTransactionState *common, Error **errp)

{

    BlockdevBackupState *state = DO_UPCAST(BlockdevBackupState, common, common);

    BlockdevBackup *backup;

    BlockBackend *blk, *target;

    Error *local_err = NULL;



    assert(common->action->kind == TRANSACTION_ACTION_KIND_BLOCKDEV_BACKUP);

    backup = common->action->blockdev_backup;



    blk = blk_by_name(backup->device);

    if (!blk) {

        error_setg(errp, "Device '%s' not found", backup->device);

        return;

    }



    if (!blk_is_available(blk)) {

        error_setg(errp, QERR_DEVICE_HAS_NO_MEDIUM, backup->device);

        return;

    }



    target = blk_by_name(backup->target);

    if (!target) {

        error_setg(errp, "Device '%s' not found", backup->target);

        return;

    }



    /* AioContext is released in .clean() */

    state->aio_context = blk_get_aio_context(blk);

    if (state->aio_context != blk_get_aio_context(target)) {

        state->aio_context = NULL;

        error_setg(errp, "Backup between two IO threads is not implemented");

        return;

    }

    aio_context_acquire(state->aio_context);

    state->bs = blk_bs(blk);

    bdrv_drained_begin(state->bs);



    qmp_blockdev_backup(backup->device, backup->target,

                        backup->sync,

                        backup->has_speed, backup->speed,

                        backup->has_on_source_error, backup->on_source_error,

                        backup->has_on_target_error, backup->on_target_error,

                        &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        return;

    }



    state->job = state->bs->job;

}
