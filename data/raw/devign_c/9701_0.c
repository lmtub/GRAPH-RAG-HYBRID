int bdrv_img_create(const char *filename, const char *fmt,

                    const char *base_filename, const char *base_fmt,

                    char *options, uint64_t img_size, int flags)

{

    QEMUOptionParameter *param = NULL, *create_options = NULL;

    QEMUOptionParameter *backing_fmt;

    BlockDriverState *bs = NULL;

    BlockDriver *drv, *proto_drv;

    int ret = 0;



    /* Find driver and parse its options */

    drv = bdrv_find_format(fmt);

    if (!drv) {

        error_report("Unknown file format '%s'", fmt);

        ret = -1;

        goto out;

    }



    proto_drv = bdrv_find_protocol(filename);

    if (!proto_drv) {

        error_report("Unknown protocol '%s'", filename);

        ret = -1;

        goto out;

    }



    create_options = append_option_parameters(create_options,

                                              drv->create_options);

    create_options = append_option_parameters(create_options,

                                              proto_drv->create_options);



    /* Create parameter list with default values */

    param = parse_option_parameters("", create_options, param);



    set_option_parameter_int(param, BLOCK_OPT_SIZE, img_size);



    /* Parse -o options */

    if (options) {

        param = parse_option_parameters(options, create_options, param);

        if (param == NULL) {

            error_report("Invalid options for file format '%s'.", fmt);

            ret = -1;

            goto out;

        }

    }



    if (base_filename) {

        if (set_option_parameter(param, BLOCK_OPT_BACKING_FILE,

                                 base_filename)) {

            error_report("Backing file not supported for file format '%s'",

                         fmt);

            ret = -1;

            goto out;

        }

    }



    if (base_fmt) {

        if (set_option_parameter(param, BLOCK_OPT_BACKING_FMT, base_fmt)) {

            error_report("Backing file format not supported for file "

                         "format '%s'", fmt);

            ret = -1;

            goto out;

        }

    }



    backing_fmt = get_option_parameter(param, BLOCK_OPT_BACKING_FMT);

    if (backing_fmt && backing_fmt->value.s) {

        if (!bdrv_find_format(backing_fmt->value.s)) {

            error_report("Unknown backing file format '%s'",

                         backing_fmt->value.s);

            ret = -1;

            goto out;

        }

    }



    // The size for the image must always be specified, with one exception:

    // If we are using a backing file, we can obtain the size from there

    if (get_option_parameter(param, BLOCK_OPT_SIZE)->value.n == -1) {

        QEMUOptionParameter *backing_file =

            get_option_parameter(param, BLOCK_OPT_BACKING_FILE);



        if (backing_file && backing_file->value.s) {

            uint64_t size;

            const char *fmt = NULL;

            char buf[32];



            if (backing_fmt && backing_fmt->value.s) {

                fmt = backing_fmt->value.s;

            }



            bs = bdrv_new("");



            ret = bdrv_open(bs, backing_file->value.s, flags, drv);

            if (ret < 0) {

                error_report("Could not open '%s'", filename);

                ret = -1;

                goto out;

            }

            bdrv_get_geometry(bs, &size);

            size *= 512;



            snprintf(buf, sizeof(buf), "%" PRId64, size);

            set_option_parameter(param, BLOCK_OPT_SIZE, buf);

        } else {

            error_report("Image creation needs a size parameter");

            ret = -1;

            goto out;

        }

    }



    printf("Formatting '%s', fmt=%s ", filename, fmt);

    print_option_parameters(param);

    puts("");



    ret = bdrv_create(drv, filename, param);



    if (ret < 0) {

        if (ret == -ENOTSUP) {

            error_report("Formatting or formatting option not supported for "

                         "file format '%s'", fmt);

        } else if (ret == -EFBIG) {

            error_report("The image size is too large for file format '%s'",

                         fmt);

        } else {

            error_report("%s: error while creating %s: %s", filename, fmt,

                         strerror(-ret));

        }

    }



out:

    free_option_parameters(create_options);

    free_option_parameters(param);



    if (bs) {

        bdrv_delete(bs);

    }

    if (ret) {

        return 1;

    }

    return 0;

}
