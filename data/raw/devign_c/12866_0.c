static BlockBackend *img_open(const char *id, const char *filename,

                              const char *fmt, int flags,

                              bool require_io, bool quiet)

{

    BlockBackend *blk;

    BlockDriverState *bs;

    char password[256];

    Error *local_err = NULL;

    QDict *options = NULL;



    if (fmt) {

        options = qdict_new();

        qdict_put(options, "driver", qstring_from_str(fmt));

    }



    blk = blk_new_open(id, filename, NULL, options, flags, &local_err);

    if (!blk) {

        error_report("Could not open '%s': %s", filename,

                     error_get_pretty(local_err));

        error_free(local_err);

        goto fail;

    }



    bs = blk_bs(blk);

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

    return blk;

fail:

    blk_unref(blk);

    return NULL;

}
