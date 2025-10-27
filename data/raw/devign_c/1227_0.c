void qmp_block_stream(const char *device, bool has_base,

                      const char *base, Error **errp)

{

    BlockDriverState *bs;

    BlockDriverState *base_bs = NULL;

    Error *local_err = NULL;



    bs = bdrv_find(device);

    if (!bs) {

        error_set(errp, QERR_DEVICE_NOT_FOUND, device);

        return;

    }



    if (base) {

        base_bs = bdrv_find_backing_image(bs, base);

        if (base_bs == NULL) {

            error_set(errp, QERR_BASE_NOT_FOUND, base);

            return;

        }

    }



    stream_start(bs, base_bs, base, block_stream_cb, bs, &local_err);

    if (error_is_set(&local_err)) {

        error_propagate(errp, local_err);

        return;

    }



    /* Grab a reference so hotplug does not delete the BlockDriverState from

     * underneath us.

     */

    drive_get_ref(drive_get_by_blockdev(bs));



    trace_qmp_block_stream(bs, bs->job);

}
