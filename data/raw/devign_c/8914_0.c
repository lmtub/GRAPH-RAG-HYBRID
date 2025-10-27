static BlockDriverState *bdrv_new_open(const char *filename,

                                       const char *fmt,

                                       int flags,

                                       bool require_io,

                                       bool quiet)

{

    BlockDriverState *bs;

    BlockDriver *drv;

    char password[256];

    Error *local_err = NULL;

    int ret;



    bs = bdrv_new("image");



    if (fmt) {

        drv = bdrv_find_format(fmt);

        if (!drv) {

            error_report("Unknown file format '%s'", fmt);

            goto fail;

        }

    } else {

        drv = NULL;

    }



    ret = bdrv_open(&bs, filename, NULL, NULL, flags, drv, &local_err);

    if (ret < 0) {

        error_report("Could not open '%s': %s", filename,

                     error_get_pretty(local_err));

        error_free(local_err);

        goto fail;

    }



    if (bdrv_is_encrypted(bs) && require_io) {

        qprintf(quiet, "Disk image '%s' is encrypted.\n", filename);

        if (read_password(password, sizeof(password)) < 0) {

            error_report("No password given");

            goto fail;

        }

        if (bdrv_set_key(bs, password) < 0) {

            error_report("invalid password");

            goto fail;

        }

    }

    return bs;

fail:

    bdrv_unref(bs);

    return NULL;

}
