static void internal_snapshot_prepare(BlkTransactionState *common,

                                      Error **errp)

{

    Error *local_err = NULL;

    const char *device;

    const char *name;

    BlockBackend *blk;

    BlockDriverState *bs;

    QEMUSnapshotInfo old_sn, *sn;

    bool ret;

    qemu_timeval tv;

    BlockdevSnapshotInternal *internal;

    InternalSnapshotState *state;

    int ret1;



    g_assert(common->action->kind ==

             TRANSACTION_ACTION_KIND_BLOCKDEV_SNAPSHOT_INTERNAL_SYNC);

    internal = common->action->blockdev_snapshot_internal_sync;

    state = DO_UPCAST(InternalSnapshotState, common, common);



    /* 1. parse input */

    device = internal->device;

    name = internal->name;



    /* 2. check for validation */

    blk = blk_by_name(device);

    if (!blk) {

        error_set(errp, ERROR_CLASS_DEVICE_NOT_FOUND,

                  "Device '%s' not found", device);

        return;

    }



    /* AioContext is released in .clean() */

    state->aio_context = blk_get_aio_context(blk);

    aio_context_acquire(state->aio_context);



    if (!blk_is_available(blk)) {

        error_setg(errp, QERR_DEVICE_HAS_NO_MEDIUM, device);

        return;

    }

    bs = blk_bs(blk);



    state->bs = bs;

    bdrv_drained_begin(bs);



    if (bdrv_op_is_blocked(bs, BLOCK_OP_TYPE_INTERNAL_SNAPSHOT, errp)) {

        return;

    }



    if (bdrv_is_read_only(bs)) {

        error_setg(errp, "Device '%s' is read only", device);

        return;

    }



    if (!bdrv_can_snapshot(bs)) {

        error_setg(errp, "Block format '%s' used by device '%s' "

                   "does not support internal snapshots",

                   bs->drv->format_name, device);

        return;

    }



    if (!strlen(name)) {

        error_setg(errp, "Name is empty");

        return;

    }



    /* check whether a snapshot with name exist */

    ret = bdrv_snapshot_find_by_id_and_name(bs, NULL, name, &old_sn,

                                            &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        return;

    } else if (ret) {

        error_setg(errp,

                   "Snapshot with name '%s' already exists on device '%s'",

                   name, device);

        return;

    }



    /* 3. take the snapshot */

    sn = &state->sn;

    pstrcpy(sn->name, sizeof(sn->name), name);

    qemu_gettimeofday(&tv);

    sn->date_sec = tv.tv_sec;

    sn->date_nsec = tv.tv_usec * 1000;

    sn->vm_clock_nsec = qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL);



    ret1 = bdrv_snapshot_create(bs, sn);

    if (ret1 < 0) {

        error_setg_errno(errp, -ret1,

                         "Failed to create snapshot '%s' on device '%s'",

                         name, device);

        return;

    }



    /* 4. succeed, mark a snapshot is created */

    state->created = true;

}
