static void do_drive_backup(DriveBackup *backup, BlockJobTxn *txn, Error **errp)

{

    BlockDriverState *bs;

    BlockDriverState *target_bs;

    BlockDriverState *source = NULL;

    BdrvDirtyBitmap *bmap = NULL;

    AioContext *aio_context;

    QDict *options = NULL;

    Error *local_err = NULL;

    int flags;

    int64_t size;



    if (!backup->has_speed) {

        backup->speed = 0;

    }

    if (!backup->has_on_source_error) {

        backup->on_source_error = BLOCKDEV_ON_ERROR_REPORT;

    }

    if (!backup->has_on_target_error) {

        backup->on_target_error = BLOCKDEV_ON_ERROR_REPORT;

    }

    if (!backup->has_mode) {

        backup->mode = NEW_IMAGE_MODE_ABSOLUTE_PATHS;

    }

    if (!backup->has_job_id) {

        backup->job_id = NULL;

    }

    if (!backup->has_compress) {

        backup->compress = false;

    }



    bs = qmp_get_root_bs(backup->device, errp);

    if (!bs) {

        return;

    }



    aio_context = bdrv_get_aio_context(bs);

    aio_context_acquire(aio_context);



    if (!backup->has_format) {

        backup->format = backup->mode == NEW_IMAGE_MODE_EXISTING ?

                         NULL : (char*) bs->drv->format_name;

    }



    /* Early check to avoid creating target */

    if (bdrv_op_is_blocked(bs, BLOCK_OP_TYPE_BACKUP_SOURCE, errp)) {

        goto out;

    }



    flags = bs->open_flags | BDRV_O_RDWR;



    /* See if we have a backing HD we can use to create our new image

     * on top of. */

    if (backup->sync == MIRROR_SYNC_MODE_TOP) {

        source = backing_bs(bs);

        if (!source) {

            backup->sync = MIRROR_SYNC_MODE_FULL;

        }

    }

    if (backup->sync == MIRROR_SYNC_MODE_NONE) {

        source = bs;

    }



    size = bdrv_getlength(bs);

    if (size < 0) {

        error_setg_errno(errp, -size, "bdrv_getlength failed");

        goto out;

    }



    if (backup->mode != NEW_IMAGE_MODE_EXISTING) {

        assert(backup->format);

        if (source) {

            bdrv_img_create(backup->target, backup->format, source->filename,

                            source->drv->format_name, NULL,

                            size, flags, &local_err, false);

        } else {

            bdrv_img_create(backup->target, backup->format, NULL, NULL, NULL,

                            size, flags, &local_err, false);

        }

    }



    if (local_err) {

        error_propagate(errp, local_err);

        goto out;

    }



    if (backup->format) {

        options = qdict_new();

        qdict_put(options, "driver", qstring_from_str(backup->format));

    }



    target_bs = bdrv_open(backup->target, NULL, options, flags, errp);

    if (!target_bs) {

        goto out;

    }



    bdrv_set_aio_context(target_bs, aio_context);



    if (backup->has_bitmap) {

        bmap = bdrv_find_dirty_bitmap(bs, backup->bitmap);

        if (!bmap) {

            error_setg(errp, "Bitmap '%s' could not be found", backup->bitmap);

            bdrv_unref(target_bs);

            goto out;

        }

    }



    backup_start(backup->job_id, bs, target_bs, backup->speed, backup->sync,

                 bmap, backup->compress, backup->on_source_error,

                 backup->on_target_error, BLOCK_JOB_DEFAULT,

                 NULL, NULL, txn, &local_err);

    bdrv_unref(target_bs);

    if (local_err != NULL) {

        error_propagate(errp, local_err);

        goto out;

    }



out:

    aio_context_release(aio_context);

}
