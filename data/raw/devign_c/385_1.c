void qmp_block_commit(const char *device,

                      bool has_base, const char *base, const char *top,

                      bool has_speed, int64_t speed,

                      Error **errp)

{

    BlockDriverState *bs;

    BlockDriverState *base_bs, *top_bs;

    Error *local_err = NULL;

    /* This will be part of the QMP command, if/when the

     * BlockdevOnError change for blkmirror makes it in

     */

    BlockdevOnError on_error = BLOCKDEV_ON_ERROR_REPORT;



    /* drain all i/o before commits */

    bdrv_drain_all();



    bs = bdrv_find(device);

    if (!bs) {

        error_set(errp, QERR_DEVICE_NOT_FOUND, device);

        return;

    }

    if (base && has_base) {

        base_bs = bdrv_find_backing_image(bs, base);

    } else {

        base_bs = bdrv_find_base(bs);

    }



    if (base_bs == NULL) {

        error_set(errp, QERR_BASE_NOT_FOUND, base ? base : "NULL");

        return;

    }



    /* default top_bs is the active layer */

    top_bs = bs;



    if (top) {

        if (strcmp(bs->filename, top) != 0) {

            top_bs = bdrv_find_backing_image(bs, top);

        }

    }



    if (top_bs == NULL) {

        error_setg(errp, "Top image file %s not found", top ? top : "NULL");

        return;

    }



    commit_start(bs, base_bs, top_bs, speed, on_error, block_job_cb, bs,

                &local_err);

    if (local_err != NULL) {

        error_propagate(errp, local_err);

        return;

    }

    /* Grab a reference so hotplug does not delete the BlockDriverState from

     * underneath us.

     */

    drive_get_ref(drive_get_by_blockdev(bs));

}
