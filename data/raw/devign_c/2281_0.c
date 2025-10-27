static void do_drive_backup(const char *job_id, const char *device,

                            const char *target, bool has_format,

                            const char *format, enum MirrorSyncMode sync,

                            bool has_mode, enum NewImageMode mode,

                            bool has_speed, int64_t speed,

                            bool has_bitmap, const char *bitmap,

                            bool has_on_source_error,

                            BlockdevOnError on_source_error,

                            bool has_on_target_error,

                            BlockdevOnError on_target_error,

                            BlockJobTxn *txn, Error **errp)

{

    BlockBackend *blk;

    BlockDriverState *bs;

    BlockDriverState *target_bs;

    BlockDriverState *source = NULL;

    BdrvDirtyBitmap *bmap = NULL;

    AioContext *aio_context;

    QDict *options = NULL;

    Error *local_err = NULL;

    int flags;

    int64_t size;



    if (!has_speed) {

        speed = 0;

    }

    if (!has_on_source_error) {

        on_source_error = BLOCKDEV_ON_ERROR_REPORT;

    }

    if (!has_on_target_error) {

        on_target_error = BLOCKDEV_ON_ERROR_REPORT;

    }

    if (!has_mode) {

        mode = NEW_IMAGE_MODE_ABSOLUTE_PATHS;

    }



    blk = blk_by_name(device);

    if (!blk) {

        error_set(errp, ERROR_CLASS_DEVICE_NOT_FOUND,

                  "Device '%s' not found", device);

        return;

    }



    aio_context = blk_get_aio_context(blk);

    aio_context_acquire(aio_context);



    /* Although backup_run has this check too, we need to use bs->drv below, so

     * do an early check redundantly. */

    if (!blk_is_available(blk)) {

        error_setg(errp, QERR_DEVICE_HAS_NO_MEDIUM, device);

        goto out;

    }

    bs = blk_bs(blk);



    if (!has_format) {

        format = mode == NEW_IMAGE_MODE_EXISTING ? NULL : bs->drv->format_name;

    }



    /* Early check to avoid creating target */

    if (bdrv_op_is_blocked(bs, BLOCK_OP_TYPE_BACKUP_SOURCE, errp)) {

        goto out;

    }



    flags = bs->open_flags | BDRV_O_RDWR;



    /* See if we have a backing HD we can use to create our new image

     * on top of. */

    if (sync == MIRROR_SYNC_MODE_TOP) {

        source = backing_bs(bs);

        if (!source) {

            sync = MIRROR_SYNC_MODE_FULL;

        }

    }

    if (sync == MIRROR_SYNC_MODE_NONE) {

        source = bs;

    }



    size = bdrv_getlength(bs);

    if (size < 0) {

        error_setg_errno(errp, -size, "bdrv_getlength failed");

        goto out;

    }



    if (mode != NEW_IMAGE_MODE_EXISTING) {

        assert(format);

        if (source) {

            bdrv_img_create(target, format, source->filename,

                            source->drv->format_name, NULL,

                            size, flags, &local_err, false);

        } else {

            bdrv_img_create(target, format, NULL, NULL, NULL,

                            size, flags, &local_err, false);

        }

    }



    if (local_err) {

        error_propagate(errp, local_err);

        goto out;

    }



    if (format) {

        options = qdict_new();

        qdict_put(options, "driver", qstring_from_str(format));

    }



    target_bs = bdrv_open(target, NULL, options, flags, errp);

    if (!target_bs) {

        goto out;

    }



    bdrv_set_aio_context(target_bs, aio_context);



    if (has_bitmap) {

        bmap = bdrv_find_dirty_bitmap(bs, bitmap);

        if (!bmap) {

            error_setg(errp, "Bitmap '%s' could not be found", bitmap);

            bdrv_unref(target_bs);

            goto out;

        }

    }



    backup_start(job_id, bs, target_bs, speed, sync, bmap,

                 on_source_error, on_target_error,

                 block_job_cb, bs, txn, &local_err);

    bdrv_unref(target_bs);

    if (local_err != NULL) {

        error_propagate(errp, local_err);

        goto out;

    }



out:

    aio_context_release(aio_context);

}
