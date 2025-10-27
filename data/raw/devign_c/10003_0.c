void qmp_drive_mirror(DriveMirror *arg, Error **errp)

{

    BlockDriverState *bs;

    BlockBackend *blk;

    BlockDriverState *source, *target_bs;

    AioContext *aio_context;

    BlockMirrorBackingMode backing_mode;

    Error *local_err = NULL;

    QDict *options = NULL;

    int flags;

    int64_t size;

    const char *format = arg->format;



    blk = blk_by_name(arg->device);

    if (!blk) {

        error_set(errp, ERROR_CLASS_DEVICE_NOT_FOUND,

                  "Device '%s' not found", arg->device);

        return;

    }



    aio_context = blk_get_aio_context(blk);

    aio_context_acquire(aio_context);



    if (!blk_is_available(blk)) {

        error_setg(errp, QERR_DEVICE_HAS_NO_MEDIUM, arg->device);

        goto out;

    }

    bs = blk_bs(blk);

    if (!arg->has_mode) {

        arg->mode = NEW_IMAGE_MODE_ABSOLUTE_PATHS;

    }



    if (!arg->has_format) {

        format = (arg->mode == NEW_IMAGE_MODE_EXISTING

                  ? NULL : bs->drv->format_name);

    }



    flags = bs->open_flags | BDRV_O_RDWR;

    source = backing_bs(bs);

    if (!source && arg->sync == MIRROR_SYNC_MODE_TOP) {

        arg->sync = MIRROR_SYNC_MODE_FULL;

    }

    if (arg->sync == MIRROR_SYNC_MODE_NONE) {

        source = bs;

    }



    size = bdrv_getlength(bs);

    if (size < 0) {

        error_setg_errno(errp, -size, "bdrv_getlength failed");

        goto out;

    }



    if (arg->has_replaces) {

        BlockDriverState *to_replace_bs;

        AioContext *replace_aio_context;

        int64_t replace_size;



        if (!arg->has_node_name) {

            error_setg(errp, "a node-name must be provided when replacing a"

                             " named node of the graph");

            goto out;

        }



        to_replace_bs = check_to_replace_node(bs, arg->replaces, &local_err);



        if (!to_replace_bs) {

            error_propagate(errp, local_err);

            goto out;

        }



        replace_aio_context = bdrv_get_aio_context(to_replace_bs);

        aio_context_acquire(replace_aio_context);

        replace_size = bdrv_getlength(to_replace_bs);

        aio_context_release(replace_aio_context);



        if (size != replace_size) {

            error_setg(errp, "cannot replace image with a mirror image of "

                             "different size");

            goto out;

        }

    }



    if (arg->mode == NEW_IMAGE_MODE_ABSOLUTE_PATHS) {

        backing_mode = MIRROR_SOURCE_BACKING_CHAIN;

    } else {

        backing_mode = MIRROR_OPEN_BACKING_CHAIN;

    }



    if ((arg->sync == MIRROR_SYNC_MODE_FULL || !source)

        && arg->mode != NEW_IMAGE_MODE_EXISTING)

    {

        /* create new image w/o backing file */

        assert(format);

        bdrv_img_create(arg->target, format,

                        NULL, NULL, NULL, size, flags, &local_err, false);

    } else {

        switch (arg->mode) {

        case NEW_IMAGE_MODE_EXISTING:

            break;

        case NEW_IMAGE_MODE_ABSOLUTE_PATHS:

            /* create new image with backing file */

            bdrv_img_create(arg->target, format,

                            source->filename,

                            source->drv->format_name,

                            NULL, size, flags, &local_err, false);

            break;

        default:

            abort();

        }

    }



    if (local_err) {

        error_propagate(errp, local_err);

        goto out;

    }



    options = qdict_new();

    if (arg->has_node_name) {

        qdict_put(options, "node-name", qstring_from_str(arg->node_name));

    }

    if (format) {

        qdict_put(options, "driver", qstring_from_str(format));

    }



    /* Mirroring takes care of copy-on-write using the source's backing

     * file.

     */

    target_bs = bdrv_open(arg->target, NULL, options,

                          flags | BDRV_O_NO_BACKING, errp);

    if (!target_bs) {

        goto out;

    }



    bdrv_set_aio_context(target_bs, aio_context);



    blockdev_mirror_common(arg->has_job_id ? arg->job_id : NULL, bs, target_bs,

                           arg->has_replaces, arg->replaces, arg->sync,

                           backing_mode, arg->has_speed, arg->speed,

                           arg->has_granularity, arg->granularity,

                           arg->has_buf_size, arg->buf_size,

                           arg->has_on_source_error, arg->on_source_error,

                           arg->has_on_target_error, arg->on_target_error,

                           arg->has_unmap, arg->unmap,

                           &local_err);

    bdrv_unref(target_bs);

    error_propagate(errp, local_err);

out:

    aio_context_release(aio_context);

}
