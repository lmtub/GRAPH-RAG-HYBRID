static int img_dd(int argc, char **argv)

{

    int ret = 0;

    char *arg = NULL;

    char *tmp;

    BlockDriver *drv = NULL, *proto_drv = NULL;

    BlockBackend *blk1 = NULL, *blk2 = NULL;

    QemuOpts *opts = NULL;

    QemuOptsList *create_opts = NULL;

    Error *local_err = NULL;

    bool image_opts = false;

    int c, i;

    const char *out_fmt = "raw";

    const char *fmt = NULL;

    int64_t size = 0;

    int64_t block_count = 0, out_pos, in_pos;

    struct DdInfo dd = {

        .flags = 0,

        .count = 0,

    };

    struct DdIo in = {

        .bsz = 512, /* Block size is by default 512 bytes */

        .filename = NULL,

        .buf = NULL,

        .offset = 0

    };

    struct DdIo out = {

        .bsz = 512,

        .filename = NULL,

        .buf = NULL,

        .offset = 0

    };



    const struct DdOpts options[] = {

        { "bs", img_dd_bs, C_BS },

        { "count", img_dd_count, C_COUNT },

        { "if", img_dd_if, C_IF },

        { "of", img_dd_of, C_OF },

        { "skip", img_dd_skip, C_SKIP },

        { NULL, NULL, 0 }

    };

    const struct option long_options[] = {

        { "help", no_argument, 0, 'h'},

        { "image-opts", no_argument, 0, OPTION_IMAGE_OPTS},

        { 0, 0, 0, 0 }

    };



    while ((c = getopt_long(argc, argv, "hf:O:", long_options, NULL))) {

        if (c == EOF) {

            break;

        }

        switch (c) {

        case 'O':

            out_fmt = optarg;

            break;

        case 'f':

            fmt = optarg;

            break;

        case '?':

            error_report("Try 'qemu-img --help' for more information.");

            ret = -1;

            goto out;

        case 'h':

            help();

            break;

        case OPTION_IMAGE_OPTS:

            image_opts = true;

            break;

        }

    }



    for (i = optind; i < argc; i++) {

        int j;

        arg = g_strdup(argv[i]);



        tmp = strchr(arg, '=');

        if (tmp == NULL) {

            error_report("unrecognized operand %s", arg);

            ret = -1;

            goto out;

        }



        *tmp++ = '\0';



        for (j = 0; options[j].name != NULL; j++) {

            if (!strcmp(arg, options[j].name)) {

                break;

            }

        }

        if (options[j].name == NULL) {

            error_report("unrecognized operand %s", arg);

            ret = -1;

            goto out;

        }



        if (options[j].f(tmp, &in, &out, &dd) != 0) {

            ret = -1;

            goto out;

        }

        dd.flags |= options[j].flag;

        g_free(arg);

        arg = NULL;

    }



    if (!(dd.flags & C_IF && dd.flags & C_OF)) {

        error_report("Must specify both input and output files");

        ret = -1;

        goto out;

    }

    blk1 = img_open(image_opts, in.filename, fmt, 0, false, false);



    if (!blk1) {

        ret = -1;

        goto out;

    }



    drv = bdrv_find_format(out_fmt);

    if (!drv) {

        error_report("Unknown file format");

        ret = -1;

        goto out;

    }

    proto_drv = bdrv_find_protocol(out.filename, true, &local_err);



    if (!proto_drv) {

        error_report_err(local_err);

        ret = -1;

        goto out;

    }

    if (!drv->create_opts) {

        error_report("Format driver '%s' does not support image creation",

                     drv->format_name);

        ret = -1;

        goto out;

    }

    if (!proto_drv->create_opts) {

        error_report("Protocol driver '%s' does not support image creation",

                     proto_drv->format_name);

        ret = -1;

        goto out;

    }

    create_opts = qemu_opts_append(create_opts, drv->create_opts);

    create_opts = qemu_opts_append(create_opts, proto_drv->create_opts);



    opts = qemu_opts_create(create_opts, NULL, 0, &error_abort);



    size = blk_getlength(blk1);

    if (size < 0) {

        error_report("Failed to get size for '%s'", in.filename);

        ret = -1;

        goto out;

    }



    if (dd.flags & C_COUNT && dd.count <= INT64_MAX / in.bsz &&

        dd.count * in.bsz < size) {

        size = dd.count * in.bsz;

    }



    /* Overflow means the specified offset is beyond input image's size */

    if (dd.flags & C_SKIP && (in.offset > INT64_MAX / in.bsz ||

                              size < in.bsz * in.offset)) {

        qemu_opt_set_number(opts, BLOCK_OPT_SIZE, 0, &error_abort);

    } else {

        qemu_opt_set_number(opts, BLOCK_OPT_SIZE,

                            size - in.bsz * in.offset, &error_abort);

    }



    ret = bdrv_create(drv, out.filename, opts, &local_err);

    if (ret < 0) {

        error_reportf_err(local_err,

                          "%s: error while creating output image: ",

                          out.filename);

        ret = -1;

        goto out;

    }



    blk2 = img_open(image_opts, out.filename, out_fmt, BDRV_O_RDWR,

                    false, false);



    if (!blk2) {

        ret = -1;

        goto out;

    }



    if (dd.flags & C_SKIP && (in.offset > INT64_MAX / in.bsz ||

                              size < in.offset * in.bsz)) {

        /* We give a warning if the skip option is bigger than the input

         * size and create an empty output disk image (i.e. like dd(1)).

         */

        error_report("%s: cannot skip to specified offset", in.filename);

        in_pos = size;

    } else {

        in_pos = in.offset * in.bsz;

    }



    in.buf = g_new(uint8_t, in.bsz);



    for (out_pos = 0; in_pos < size; block_count++) {

        int in_ret, out_ret;



        if (in_pos + in.bsz > size) {

            in_ret = blk_pread(blk1, in_pos, in.buf, size - in_pos);

        } else {

            in_ret = blk_pread(blk1, in_pos, in.buf, in.bsz);

        }

        if (in_ret < 0) {

            error_report("error while reading from input image file: %s",

                         strerror(-in_ret));

            ret = -1;

            goto out;

        }

        in_pos += in_ret;



        out_ret = blk_pwrite(blk2, out_pos, in.buf, in_ret, 0);



        if (out_ret < 0) {

            error_report("error while writing to output image file: %s",

                         strerror(-out_ret));

            ret = -1;

            goto out;

        }

        out_pos += out_ret;

    }



out:

    g_free(arg);

    qemu_opts_del(opts);

    qemu_opts_free(create_opts);

    blk_unref(blk1);

    blk_unref(blk2);

    g_free(in.filename);

    g_free(out.filename);

    g_free(in.buf);

    g_free(out.buf);



    if (ret) {

        return 1;

    }

    return 0;

}
