BlockJob *backup_job_create(const char *job_id, BlockDriverState *bs,

                  BlockDriverState *target, int64_t speed,

                  MirrorSyncMode sync_mode, BdrvDirtyBitmap *sync_bitmap,

                  bool compress,

                  BlockdevOnError on_source_error,

                  BlockdevOnError on_target_error,

                  int creation_flags,

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

        return NULL;

    }



    if (!bdrv_is_inserted(bs)) {

        error_setg(errp, "Device is not inserted: %s",

                   bdrv_get_device_name(bs));

        return NULL;

    }



    if (!bdrv_is_inserted(target)) {

        error_setg(errp, "Device is not inserted: %s",

                   bdrv_get_device_name(target));

        return NULL;

    }



    if (compress && target->drv->bdrv_co_pwritev_compressed == NULL) {

        error_setg(errp, "Compression is not supported for this drive %s",

                   bdrv_get_device_name(target));

        return NULL;

    }



    if (bdrv_op_is_blocked(bs, BLOCK_OP_TYPE_BACKUP_SOURCE, errp)) {

        return NULL;

    }



    if (bdrv_op_is_blocked(target, BLOCK_OP_TYPE_BACKUP_TARGET, errp)) {

        return NULL;

    }



    if (sync_mode == MIRROR_SYNC_MODE_INCREMENTAL) {

        if (!sync_bitmap) {

            error_setg(errp, "must provide a valid bitmap name for "

                             "\"incremental\" sync mode");

            return NULL;

        }



        /* Create a new bitmap, and freeze/disable this one. */

        if (bdrv_dirty_bitmap_create_successor(bs, sync_bitmap, errp) < 0) {

            return NULL;

        }

    } else if (sync_bitmap) {

        error_setg(errp,

                   "a sync_bitmap was provided to backup_run, "

                   "but received an incompatible sync_mode (%s)",

                   MirrorSyncMode_lookup[sync_mode]);

        return NULL;

    }



    len = bdrv_getlength(bs);

    if (len < 0) {

        error_setg_errno(errp, -len, "unable to get length for '%s'",

                         bdrv_get_device_name(bs));

        goto error;

    }



    /* job->common.len is fixed, so we can't allow resize */

    job = block_job_create(job_id, &backup_job_driver, bs,

                           BLK_PERM_CONSISTENT_READ,

                           BLK_PERM_CONSISTENT_READ | BLK_PERM_WRITE |

                           BLK_PERM_WRITE_UNCHANGED | BLK_PERM_GRAPH_MOD,

                           speed, creation_flags, cb, opaque, errp);

    if (!job) {

        goto error;

    }



    /* The target must match the source in size, so no resize here either */

    job->target = blk_new(BLK_PERM_WRITE,

                          BLK_PERM_CONSISTENT_READ | BLK_PERM_WRITE |

                          BLK_PERM_WRITE_UNCHANGED | BLK_PERM_GRAPH_MOD);

    ret = blk_insert_bs(job->target, target, errp);

    if (ret < 0) {

        goto error;

    }



    job->on_source_error = on_source_error;

    job->on_target_error = on_target_error;

    job->sync_mode = sync_mode;

    job->sync_bitmap = sync_mode == MIRROR_SYNC_MODE_INCREMENTAL ?

                       sync_bitmap : NULL;

    job->compress = compress;



    /* If there is no backing file on the target, we cannot rely on COW if our

     * backup cluster size is smaller than the target cluster size. Even for

     * targets with a backing file, try to avoid COW if possible. */

    ret = bdrv_get_info(target, &bdi);

    if (ret == -ENOTSUP && !target->backing) {

        /* Cluster size is not defined */

        error_report("WARNING: The target block device doesn't provide "

                     "information about the block size and it doesn't have a "

                     "backing file. The default block size of %u bytes is "

                     "used. If the actual block size of the target exceeds "

                     "this default, the backup may be unusable",

                     BACKUP_CLUSTER_SIZE_DEFAULT);

        job->cluster_size = BACKUP_CLUSTER_SIZE_DEFAULT;

    } else if (ret < 0 && !target->backing) {

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



    /* Required permissions are already taken with target's blk_new() */

    block_job_add_bdrv(&job->common, "target", target, 0, BLK_PERM_ALL,

                       &error_abort);

    job->common.len = len;

    block_job_txn_add_job(txn, &job->common);



    return &job->common;



 error:

    if (sync_bitmap) {

        bdrv_reclaim_dirty_bitmap(bs, sync_bitmap, NULL);

    }

    if (job) {

        backup_clean(&job->common);

        block_job_early_fail(&job->common);

    }



    return NULL;

}
