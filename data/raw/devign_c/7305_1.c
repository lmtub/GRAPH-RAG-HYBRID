int bdrv_open(BlockDriverState *bs, const char *filename, QDict *options,

              int flags, BlockDriver *drv, Error **errp)

{

    int ret;

    /* TODO: extra byte is a hack to ensure MAX_PATH space on Windows. */

    char tmp_filename[PATH_MAX + 1];

    BlockDriverState *file = NULL;

    QDict *file_options = NULL;

    const char *drvname;

    Error *local_err = NULL;



    /* NULL means an empty set of options */

    if (options == NULL) {

        options = qdict_new();

    }



    bs->options = options;

    options = qdict_clone_shallow(options);



    /* For snapshot=on, create a temporary qcow2 overlay */

    if (flags & BDRV_O_SNAPSHOT) {

        BlockDriverState *bs1;

        int64_t total_size;

        BlockDriver *bdrv_qcow2;

        QEMUOptionParameter *create_options;

        char backing_filename[PATH_MAX];



        if (qdict_size(options) != 0) {

            error_setg(errp, "Can't use snapshot=on with driver-specific options");

            ret = -EINVAL;

            goto fail;

        }

        assert(filename != NULL);



        /* if snapshot, we create a temporary backing file and open it

           instead of opening 'filename' directly */



        /* if there is a backing file, use it */

        bs1 = bdrv_new("");

        ret = bdrv_open(bs1, filename, NULL, 0, drv, &local_err);

        if (ret < 0) {

            bdrv_unref(bs1);

            goto fail;

        }

        total_size = bdrv_getlength(bs1) & BDRV_SECTOR_MASK;



        bdrv_unref(bs1);



        ret = get_tmp_filename(tmp_filename, sizeof(tmp_filename));

        if (ret < 0) {

            error_setg_errno(errp, -ret, "Could not get temporary filename");

            goto fail;

        }



        /* Real path is meaningless for protocols */

        if (path_has_protocol(filename)) {

            snprintf(backing_filename, sizeof(backing_filename),

                     "%s", filename);

        } else if (!realpath(filename, backing_filename)) {

            error_setg_errno(errp, errno, "Could not resolve path '%s'", filename);

            ret = -errno;

            goto fail;

        }



        bdrv_qcow2 = bdrv_find_format("qcow2");

        create_options = parse_option_parameters("", bdrv_qcow2->create_options,

                                                 NULL);



        set_option_parameter_int(create_options, BLOCK_OPT_SIZE, total_size);

        set_option_parameter(create_options, BLOCK_OPT_BACKING_FILE,

                             backing_filename);

        if (drv) {

            set_option_parameter(create_options, BLOCK_OPT_BACKING_FMT,

                drv->format_name);

        }



        ret = bdrv_create(bdrv_qcow2, tmp_filename, create_options, &local_err);

        free_option_parameters(create_options);

        if (ret < 0) {

            error_setg_errno(errp, -ret, "Could not create temporary overlay "

                             "'%s': %s", tmp_filename,

                             error_get_pretty(local_err));

            error_free(local_err);

            local_err = NULL;

            goto fail;

        }



        filename = tmp_filename;

        drv = bdrv_qcow2;

        bs->is_temporary = 1;

    }



    /* Open image file without format layer */

    if (flags & BDRV_O_RDWR) {

        flags |= BDRV_O_ALLOW_RDWR;

    }



    qdict_extract_subqdict(options, &file_options, "file.");



    ret = bdrv_file_open(&file, filename, file_options,

                         bdrv_open_flags(bs, flags | BDRV_O_UNMAP), &local_err);

    if (ret < 0) {

        goto fail;

    }



    /* Find the right image format driver */

    drvname = qdict_get_try_str(options, "driver");

    if (drvname) {

        drv = bdrv_find_whitelisted_format(drvname, !(flags & BDRV_O_RDWR));

        qdict_del(options, "driver");

    }



    if (!drv) {

        ret = find_image_format(file, filename, &drv, &local_err);

    }



    if (!drv) {

        goto unlink_and_fail;

    }



    /* Open the image */

    ret = bdrv_open_common(bs, file, options, flags, drv, &local_err);

    if (ret < 0) {

        goto unlink_and_fail;

    }



    if (bs->file != file) {

        bdrv_unref(file);

        file = NULL;

    }



    /* If there is a backing file, use it */

    if ((flags & BDRV_O_NO_BACKING) == 0) {

        QDict *backing_options;



        qdict_extract_subqdict(options, &backing_options, "backing.");

        ret = bdrv_open_backing_file(bs, backing_options, &local_err);

        if (ret < 0) {

            goto close_and_fail;

        }

    }



    /* Check if any unknown options were used */

    if (qdict_size(options) != 0) {

        const QDictEntry *entry = qdict_first(options);

        error_setg(errp, "Block format '%s' used by device '%s' doesn't "

                   "support the option '%s'", drv->format_name, bs->device_name,

                   entry->key);



        ret = -EINVAL;

        goto close_and_fail;

    }

    QDECREF(options);



    if (!bdrv_key_required(bs)) {

        bdrv_dev_change_media_cb(bs, true);

    }



    return 0;



unlink_and_fail:

    if (file != NULL) {

        bdrv_unref(file);

    }

    if (bs->is_temporary) {

        unlink(filename);

    }

fail:

    QDECREF(bs->options);

    QDECREF(options);

    bs->options = NULL;

    if (error_is_set(&local_err)) {

        error_propagate(errp, local_err);

    }

    return ret;



close_and_fail:

    bdrv_close(bs);

    QDECREF(options);

    if (error_is_set(&local_err)) {

        error_propagate(errp, local_err);

    }

    return ret;

}
