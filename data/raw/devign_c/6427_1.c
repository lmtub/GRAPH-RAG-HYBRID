void backup_start(const char *job_id, BlockDriverState *bs,

                  BlockDriverState *target, int64_t speed,

                  MirrorSyncMode sync_mode, BdrvDirtyBitmap *sync_bitmap,

                  BlockdevOnError on_source_error,

                  BlockdevOnError on_target_error,

                  BlockCompletionFunc *cb, void *opaque,

                  BlockJobTxn *txn, Error **errp)

{

    int64_t len;

    BlockDriverInfo bdi;

    BackupBlockJob *job = NULL;

    int ret;



    assert(bs);

    assert(target);



    if (bs == target) {

        error_setg(errp, "Source and target cannot be the same");

        return;

    }



    if (!bdrv_is_inserted(bs)) {

        error_setg(errp, "Device is not inserted: %s",

                   bdrv_get_device_name(bs));

        return;

    }



    if (!bdrv_is_inserted(target)) {

        error_setg(errp, "Device is not inserted: %s",

                   bdrv_get_device_name(target));

        return;

    }



    if (bdrv_op_is_blocked(bs, BLOCK_OP_TYPE_BACKUP_SOURCE, errp)) {

        return;

    }



    if (bdrv_op_is_blocked(target, BLOCK_OP_TYPE_BACKUP_TARGET, errp)) {

        return;

    }



    if (sync_mode == MIRROR_SYNC_MODE_INCREMENTAL) {

        if (!sync_bitmap) {

            error_setg(errp, "must provide a valid bitmap name for "

                             "\"incremental\" sync mode");

            return;

        }



        /* Create a new bitmap, and freeze/disable this one. */

        if (bdrv_dirty_bitmap_create_successor(bs, sync_bitmap, errp) < 0) {

            return;

        }

    } else if (sync_bitmap) {

        error_setg(errp,

                   "a sync_bitmap was provided to backup_run, "

                   "but received an incompatible sync_mode (%s)",

                   MirrorSyncMode_lookup[sync_mode]);

        return;

    }



    len = bdrv_getlength(bs);

    if (len < 0) {

        error_setg_errno(errp, -len, "unable to get length for '%s'",

                         bdrv_get_device_name(bs));

        goto error;

    }



    job = block_job_create(job_id, &backup_job_driver, bs, speed,

                           cb, opaque, errp);

    if (!job) {

        goto error;

    }



    job->target = blk_new();

    blk_insert_bs(job->target, target);



    job->on_source_error = on_source_error;

    job->on_target_error = on_target_error;

    job->sync_mode = sync_mode;

    job->sync_bitmap = sync_mode == MIRROR_SYNC_MODE_INCREMENTAL ?

                       sync_bitmap : NULL;



    /* If there is no backing file on the target, we cannot rely on COW if our

     * backup cluster size is smaller than the target cluster size. Even for

     * targets with a backing file, try to avoid COW if possible. */

    ret = bdrv_get_info(target, &bdi);

    if (ret < 0 && !target->backing) {

        error_setg_errno(errp, -ret,

            "Couldn't determine the cluster size of the target image, "

            "which has no backing file");

        error_append_hint(errp,

            "Aborting, since this may create an unusable destination image\n");

        goto error;

    } else if (ret < 0 && target->backing) {

        /* Not fatal; just trudge on ahead. */

        job->cluster_size = BACKUP_CLUSTER_SIZE_DEFAULT;

    } else {

        job->cluster_size = MAX(BACKUP_CLUSTER_SIZE_DEFAULT, bdi.cluster_size);

    }



    bdrv_op_block_all(target, job->common.blocker);

    job->common.len = len;

    job->common.co = qemu_coroutine_create(backup_run);

    block_job_txn_add_job(txn, &job->common);

    qemu_coroutine_enter(job->common.co, job);

    return;



 error:

    if (sync_bitmap) {

        bdrv_reclaim_dirty_bitmap(bs, sync_bitmap, NULL);

    }

    if (job) {

        blk_unref(job->target);

        block_job_unref(&job->common);

    }

}
