static int blkdebug_open(BlockDriverState *bs, QDict *options, int flags,

                         Error **errp)

{

    BDRVBlkdebugState *s = bs->opaque;

    QemuOpts *opts;

    Error *local_err = NULL;

    int ret;



    opts = qemu_opts_create(&runtime_opts, NULL, 0, &error_abort);

    qemu_opts_absorb_qdict(opts, options, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        ret = -EINVAL;

        goto out;

    }



    /* Read rules from config file or command line options */

    s->config_file = g_strdup(qemu_opt_get(opts, "config"));

    ret = read_config(s, s->config_file, options, errp);

    if (ret) {

        goto out;

    }



    /* Set initial state */

    s->state = 1;



    /* Open the image file */

    bs->file = bdrv_open_child(qemu_opt_get(opts, "x-image"), options, "image",

                               bs, &child_file, false, &local_err);

    if (local_err) {

        ret = -EINVAL;

        error_propagate(errp, local_err);

        goto out;

    }



    bs->supported_write_flags = BDRV_REQ_FUA &

        bs->file->bs->supported_write_flags;

    bs->supported_zero_flags = (BDRV_REQ_FUA | BDRV_REQ_MAY_UNMAP) &

        bs->file->bs->supported_zero_flags;

    ret = -EINVAL;



    /* Set request alignment */

    s->align = qemu_opt_get_size(opts, "align", 0);

    if (s->align && (s->align >= INT_MAX || !is_power_of_2(s->align))) {

        error_setg(errp, "Cannot meet constraints with align %" PRIu64,

                   s->align);

        goto out;

    }



    ret = 0;

out:

    if (ret < 0) {

        g_free(s->config_file);

    }

    qemu_opts_del(opts);

    return ret;

}
