int bdrv_open(BlockDriverState *bs, const char *filename, QDict *options,

              int flags, BlockDriver *drv)

{

    int ret;

    /* TODO: extra byte is a hack to ensure MAX_PATH space on Windows. */

    char tmp_filename[PATH_MAX + 1];

    BlockDriverState *file = NULL;

    QDict *file_options = NULL;



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

            error_report("Can't use snapshot=on with driver-specific options");

            ret = -EINVAL;

            goto fail;

        }

        assert(filename != NULL);



        /* if snapshot, we create a temporary backing file and open it

           instead of opening 'filename' directly */



        /* if there is a backing file, use it */

        bs1 = bdrv_new("");

        ret = bdrv_open(bs1, filename, NULL, 0, drv);

        if (ret < 0) {

            bdrv_delete(bs1);

            goto fail;

        }

        total_size = bdrv_getlength(bs1) & BDRV_SECTOR_MASK;



        bdrv_delete(bs1);



        ret = get_tmp_filename(tmp_filename, sizeof(tmp_filename));

        if (ret < 0) {

            goto fail;

        }



        /* Real path is meaningless for protocols */

        if (path_has_protocol(filename)) {

            snprintf(backing_filename, sizeof(backing_filename),

                     "%s", filename);

        } else if (!realpath(filename, backing_filename)) {

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



        ret = bdrv_create(bdrv_qcow2, tmp_filename, create_options);

        free_option_parameters(create_options);

        if (ret < 0) {

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



    extract_subqdict(options, &file_options, "file.");



    ret = bdrv_file_open(&file, filename, file_options,

                         bdrv_open_flags(bs, flags));

    if (ret < 0) {

        goto fail;

    }



    /* Find the right image format driver */

    if (!drv) {

        ret = find_image_format(file, filename, &drv);

    }



    if (!drv) {

        goto unlink_and_fail;

    }



    /* Open the image */

    ret = bdrv_open_common(bs, file, filename, options, flags, drv);

    if (ret < 0) {

        goto unlink_and_fail;

    }



    if (bs->file != file) {

        bdrv_delete(file);

        file = NULL;

    }



    /* If there is a backing file, use it */

    if ((flags & BDRV_O_NO_BACKING) == 0) {

        ret = bdrv_open_backing_file(bs);

        if (ret < 0) {

            goto close_and_fail;

        }

    }



    /* Check if any unknown options were used */

    if (qdict_size(options) != 0) {

        const QDictEntry *entry = qdict_first(options);

        qerror_report(ERROR_CLASS_GENERIC_ERROR, "Block format '%s' used by "

            "device '%s' doesn't support the option '%s'",

            drv->format_name, bs->device_name, entry->key);



        ret = -EINVAL;

        goto close_and_fail;

    }

    QDECREF(options);



    if (!bdrv_key_required(bs)) {

        bdrv_dev_change_media_cb(bs, true);

    }



    /* throttling disk I/O limits */

    if (bs->io_limits_enabled) {

        bdrv_io_limits_enable(bs);

    }



    return 0;



unlink_and_fail:

    if (file != NULL) {

        bdrv_delete(file);

    }

    if (bs->is_temporary) {

        unlink(filename);

    }

fail:

    QDECREF(bs->options);

    QDECREF(options);

    bs->options = NULL;

    return ret;



close_and_fail:

    bdrv_close(bs);

    QDECREF(options);

    return ret;

}
