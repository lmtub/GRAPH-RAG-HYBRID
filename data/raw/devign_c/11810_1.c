void commit_active_start(BlockDriverState *bs, BlockDriverState *base,

                         int64_t speed,

                         BlockdevOnError on_error,

                         BlockDriverCompletionFunc *cb,

                         void *opaque, Error **errp)

{

    int64_t length, base_length;

    int orig_base_flags;



    orig_base_flags = bdrv_get_flags(base);



    if (bdrv_reopen(base, bs->open_flags, errp)) {

        return;

    }



    length = bdrv_getlength(bs);

    if (length < 0) {

        error_setg(errp, "Unable to determine length of %s", bs->filename);

        goto error_restore_flags;

    }



    base_length = bdrv_getlength(base);

    if (base_length < 0) {

        error_setg(errp, "Unable to determine length of %s", base->filename);

        goto error_restore_flags;

    }



    if (length > base_length) {

        if (bdrv_truncate(base, length) < 0) {

            error_setg(errp, "Top image %s is larger than base image %s, and "

                             "resize of base image failed",

                             bs->filename, base->filename);

            goto error_restore_flags;

        }

    }



    bdrv_ref(base);

    mirror_start_job(bs, base, speed, 0, 0,

                     on_error, on_error, cb, opaque, errp,

                     &commit_active_job_driver, false, base);

    if (error_is_set(errp)) {

        goto error_restore_flags;

    }



    return;



error_restore_flags:

    /* ignore error and errp for bdrv_reopen, because we want to propagate

     * the original error */

    bdrv_reopen(base, orig_base_flags, NULL);

    return;

}
