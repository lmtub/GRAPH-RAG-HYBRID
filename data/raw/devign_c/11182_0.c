void bdrv_append_temp_snapshot(BlockDriverState *bs, Error **errp)

{

    /* TODO: extra byte is a hack to ensure MAX_PATH space on Windows. */

    char tmp_filename[PATH_MAX + 1];



    int64_t total_size;

    BlockDriver *bdrv_qcow2;

    QEMUOptionParameter *create_options;

    QDict *snapshot_options;

    BlockDriverState *bs_snapshot;

    Error *local_err;

    int ret;



    /* if snapshot, we create a temporary backing file and open it

       instead of opening 'filename' directly */



    /* Get the required size from the image */

    total_size = bdrv_getlength(bs);

    if (total_size < 0) {

        error_setg_errno(errp, -total_size, "Could not get image size");

        return;

    }

    total_size &= BDRV_SECTOR_MASK;



    /* Create the temporary image */

    ret = get_tmp_filename(tmp_filename, sizeof(tmp_filename));

    if (ret < 0) {

        error_setg_errno(errp, -ret, "Could not get temporary filename");

        return;

    }



    bdrv_qcow2 = bdrv_find_format("qcow2");

    create_options = parse_option_parameters("", bdrv_qcow2->create_options,

                                             NULL);



    set_option_parameter_int(create_options, BLOCK_OPT_SIZE, total_size);



    ret = bdrv_create(bdrv_qcow2, tmp_filename, create_options, &local_err);

    free_option_parameters(create_options);

    if (ret < 0) {

        error_setg_errno(errp, -ret, "Could not create temporary overlay "

                         "'%s': %s", tmp_filename,

                         error_get_pretty(local_err));

        error_free(local_err);

        return;

    }



    /* Prepare a new options QDict for the temporary file */

    snapshot_options = qdict_new();

    qdict_put(snapshot_options, "file.driver",

              qstring_from_str("file"));

    qdict_put(snapshot_options, "file.filename",

              qstring_from_str(tmp_filename));



    bs_snapshot = bdrv_new("");

    bs_snapshot->is_temporary = 1;



    ret = bdrv_open(&bs_snapshot, NULL, NULL, snapshot_options,

                    bs->open_flags & ~BDRV_O_SNAPSHOT, bdrv_qcow2, &local_err);

    if (ret < 0) {

        error_propagate(errp, local_err);

        return;

    }



    bdrv_append(bs_snapshot, bs);

}
