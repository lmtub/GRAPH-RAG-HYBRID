void qmp_block_resize(bool has_device, const char *device,

                      bool has_node_name, const char *node_name,

                      int64_t size, Error **errp)

{

    Error *local_err = NULL;

    BlockDriverState *bs;

    AioContext *aio_context;

    int ret;



    bs = bdrv_lookup_bs(has_device ? device : NULL,

                        has_node_name ? node_name : NULL,

                        &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        return;

    }



    aio_context = bdrv_get_aio_context(bs);

    aio_context_acquire(aio_context);



    if (!bdrv_is_first_non_filter(bs)) {

        error_setg(errp, QERR_FEATURE_DISABLED, "resize");

        goto out;

    }



    if (size < 0) {

        error_setg(errp, QERR_INVALID_PARAMETER_VALUE, "size", "a >0 size");

        goto out;

    }



    if (bdrv_op_is_blocked(bs, BLOCK_OP_TYPE_RESIZE, NULL)) {

        error_setg(errp, QERR_DEVICE_IN_USE, device);

        goto out;

    }



    /* complete all in-flight operations before resizing the device */

    bdrv_drain_all();



    ret = bdrv_truncate(bs, size);

    switch (ret) {

    case 0:

        break;

    case -ENOMEDIUM:

        error_setg(errp, QERR_DEVICE_HAS_NO_MEDIUM, device);

        break;

    case -ENOTSUP:

        error_setg(errp, QERR_UNSUPPORTED);

        break;

    case -EACCES:

        error_setg(errp, "Device '%s' is read only", device);

        break;

    case -EBUSY:

        error_setg(errp, QERR_DEVICE_IN_USE, device);

        break;

    default:

        error_setg_errno(errp, -ret, "Could not resize");

        break;

    }



out:

    aio_context_release(aio_context);

}
