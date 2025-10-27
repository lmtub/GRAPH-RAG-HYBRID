void qmp_block_stream(const char *device,

                      bool has_base, const char *base,

                      bool has_backing_file, const char *backing_file,

                      bool has_speed, int64_t speed,

                      bool has_on_error, BlockdevOnError on_error,

                      Error **errp)

{

    BlockDriverState *bs;

    BlockDriverState *base_bs = NULL;

    Error *local_err = NULL;

    const char *base_name = NULL;



    if (!has_on_error) {

        on_error = BLOCKDEV_ON_ERROR_REPORT;

    }



    bs = bdrv_find(device);

    if (!bs) {

        error_set(errp, QERR_DEVICE_NOT_FOUND, device);

        return;

    }



    if (bdrv_op_is_blocked(bs, BLOCK_OP_TYPE_STREAM, errp)) {

        return;

    }



    if (has_base) {

        base_bs = bdrv_find_backing_image(bs, base);

        if (base_bs == NULL) {

            error_set(errp, QERR_BASE_NOT_FOUND, base);

            return;

        }

        base_name = base;

    }



    /* if we are streaming the entire chain, the result will have no backing

     * file, and specifying one is therefore an error */

    if (base_bs == NULL && has_backing_file) {

        error_setg(errp, "backing file specified, but streaming the "

                         "entire chain");

        return;

    }



    /* backing_file string overrides base bs filename */

    base_name = has_backing_file ? backing_file : base_name;



    stream_start(bs, base_bs, base_name, has_speed ? speed : 0,

                 on_error, block_job_cb, bs, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        return;

    }



    trace_qmp_block_stream(bs, bs->job);

}
