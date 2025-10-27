SnapshotInfo *qmp_blockdev_snapshot_delete_internal_sync(const char *device,

                                                         bool has_id,

                                                         const char *id,

                                                         bool has_name,

                                                         const char *name,

                                                         Error **errp)

{

    BlockDriverState *bs;

    BlockBackend *blk;

    AioContext *aio_context;

    QEMUSnapshotInfo sn;

    Error *local_err = NULL;

    SnapshotInfo *info = NULL;

    int ret;



    blk = blk_by_name(device);

    if (!blk) {

        error_set(errp, ERROR_CLASS_DEVICE_NOT_FOUND,

                  "Device '%s' not found", device);

        return NULL;

    }



    aio_context = blk_get_aio_context(blk);

    aio_context_acquire(aio_context);



    if (!has_id) {

        id = NULL;

    }



    if (!has_name) {

        name = NULL;

    }



    if (!id && !name) {

        error_setg(errp, "Name or id must be provided");

        goto out_aio_context;

    }



    if (!blk_is_available(blk)) {

        error_setg(errp, "Device '%s' has no medium", device);

        goto out_aio_context;

    }

    bs = blk_bs(blk);



    if (bdrv_op_is_blocked(bs, BLOCK_OP_TYPE_INTERNAL_SNAPSHOT_DELETE, errp)) {

        goto out_aio_context;

    }



    ret = bdrv_snapshot_find_by_id_and_name(bs, id, name, &sn, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        goto out_aio_context;

    }

    if (!ret) {

        error_setg(errp,

                   "Snapshot with id '%s' and name '%s' does not exist on "

                   "device '%s'",

                   STR_OR_NULL(id), STR_OR_NULL(name), device);

        goto out_aio_context;

    }



    bdrv_snapshot_delete(bs, id, name, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        goto out_aio_context;

    }



    aio_context_release(aio_context);



    info = g_new0(SnapshotInfo, 1);

    info->id = g_strdup(sn.id_str);

    info->name = g_strdup(sn.name);

    info->date_nsec = sn.date_nsec;

    info->date_sec = sn.date_sec;

    info->vm_state_size = sn.vm_state_size;

    info->vm_clock_nsec = sn.vm_clock_nsec % 1000000000;

    info->vm_clock_sec = sn.vm_clock_nsec / 1000000000;



    return info;



out_aio_context:

    aio_context_release(aio_context);

    return NULL;

}
