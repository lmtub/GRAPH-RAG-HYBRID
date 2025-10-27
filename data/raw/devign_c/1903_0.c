static BlockDriverState *bdrv_new_open(const char *filename,

                                       const char *fmt,

                                       int flags)

{

    BlockDriverState *bs;

    BlockDriver *drv;

    char password[256];



    bs = bdrv_new("");

    if (!bs) {

        error_report("Not enough memory");

        goto fail;

    }

    if (fmt) {

        drv = bdrv_find_format(fmt);

        if (!drv) {

            error_report("Unknown file format '%s'", fmt);

            goto fail;

        }

    } else {

        drv = NULL;

    }

    if (bdrv_open(bs, filename, flags, drv) < 0) {

        error_report("Could not open '%s'", filename);

        goto fail;

    }

    if (bdrv_is_encrypted(bs)) {

        printf("Disk image '%s' is encrypted.\n", filename);

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

    if (bs) {

        bdrv_delete(bs);

    }

    return NULL;

}
