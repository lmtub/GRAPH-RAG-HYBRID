static BlockDriverState *bdrv_append_temp_snapshot(BlockDriverState *bs,

                                                   int flags,

                                                   QDict *snapshot_options,

                                                   Error **errp)

{

    /* TODO: extra byte is a hack to ensure MAX_PATH space on Windows. */

    char *tmp_filename = g_malloc0(PATH_MAX + 1);

    int64_t total_size;

    QemuOpts *opts = NULL;

    BlockDriverState *bs_snapshot;

    Error *local_err = NULL;

    int ret;



    /* if snapshot, we create a temporary backing file and open it

       instead of opening 'filename' directly */



    /* Get the required size from the image */

    total_size = bdrv_getlength(bs);

    if (total_size < 0) {

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

    ret = bdrv_create(&bdrv_qcow2, tmp_filename, opts, errp);

    qemu_opts_del(opts);

    if (ret < 0) {

        error_prepend(errp, "Could not create temporary overlay '%s': ",

                      tmp_filename);

        goto out;

    }



    /* Prepare options QDict for the temporary file */

    qdict_put_str(snapshot_options, "file.driver", "file");

    qdict_put_str(snapshot_options, "file.filename", tmp_filename);

    qdict_put_str(snapshot_options, "driver", "qcow2");



    bs_snapshot = bdrv_open(NULL, NULL, snapshot_options, flags, errp);

    snapshot_options = NULL;

    if (!bs_snapshot) {

        ret = -EINVAL;

        goto out;

    }



    /* bdrv_append() consumes a strong reference to bs_snapshot (i.e. it will

     * call bdrv_unref() on it), so in order to be able to return one, we have

     * to increase bs_snapshot's refcount here */

    bdrv_ref(bs_snapshot);

    bdrv_append(bs_snapshot, bs, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        ret = -EINVAL;

        goto out;

    }



    g_free(tmp_filename);

    return bs_snapshot;



out:

    QDECREF(snapshot_options);

    g_free(tmp_filename);

    return NULL;

}
