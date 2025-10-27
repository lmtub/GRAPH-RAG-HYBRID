static BlockBackend *img_open_opts(const char *optstr,

                                   QemuOpts *opts, int flags, bool writethrough,

                                   bool quiet, bool force_share)

{

    QDict *options;

    Error *local_err = NULL;

    BlockBackend *blk;

    options = qemu_opts_to_qdict(opts, NULL);

    if (force_share) {

        if (qdict_haskey(options, BDRV_OPT_FORCE_SHARE)

            && !qdict_get_bool(options, BDRV_OPT_FORCE_SHARE)) {

            error_report("--force-share/-U conflicts with image options");


            return NULL;

        }

        qdict_put(options, BDRV_OPT_FORCE_SHARE, qbool_from_bool(true));

    }

    blk = blk_new_open(NULL, NULL, options, flags, &local_err);

    if (!blk) {

        error_reportf_err(local_err, "Could not open '%s': ", optstr);

        return NULL;

    }

    blk_set_enable_write_cache(blk, !writethrough);



    if (img_open_password(blk, optstr, flags, quiet) < 0) {

        blk_unref(blk);

        return NULL;

    }

    return blk;

}