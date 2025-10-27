int bdrv_append_temp_snapshot(BlockDriverState *bs, int flags, Error **errp)

{

    /* TODO: extra byte is a hack to ensure MAX_PATH space on Windows. */

    char *tmp_filename = g_malloc0(PATH_MAX + 1);

    int64_t total_size;

    QemuOpts *opts = NULL;

    QDict *snapshot_options;

    BlockDriverState *bs_snapshot;

    Error *local_err;

    int ret;



    /* if snapshot, we create a temporary backing file and open it

       instead of opening 'filename' directly */



    /* Get the required size from the image */

    total_size = bdrv_getlength(bs);

    if (total_size < 0) {

        ret = total_size;

        error_setg_errno(errp, -total_size, "Could not get image size");

        goto out;

    }



    /* Create the temporary image */

    ret = get_tmp_filename(tmp_filename, PATH_MAX + 1);

    if (ret < 0) {

        error_setg_errno(errp, -ret, "Could not get temporary filename");

        goto out;

    }



    opts = qemu_opts_create(bdrv_qcow2.create_opts, NULL, 0,

                            &error_abort);

    qemu_opt_set_number(opts, BLOCK_OPT_SIZE, total_size, &error_abort);

    ret = bdrv_create(&bdrv_qcow2, tmp_filename, opts, &local_err);

    qemu_opts_del(opts);

    if (ret < 0) {

        error_setg_errno(errp, -ret, "Could not create temporary overlay "

                         "'%s': %s", tmp_filename,

                         error_get_pretty(local_err));

        error_free(local_err);

        goto out;

    }



    /* Prepare a new options QDict for the temporary file */

    snapshot_options = qdict_new();

    qdict_put(snapshot_options, "file.driver",

              qstring_from_str("file"));

    qdict_put(snapshot_options, "file.filename",

              qstring_from_str(tmp_filename));



    bs_snapshot = bdrv_new();



    ret = bdrv_open(&bs_snapshot, NULL, NULL, snapshot_options,

                    flags, &bdrv_qcow2, &local_err);

    if (ret < 0) {

        error_propagate(errp, local_err);

        goto out;

    }



    bdrv_append(bs_snapshot, bs);



out:

    g_free(tmp_filename);

    return ret;

}
