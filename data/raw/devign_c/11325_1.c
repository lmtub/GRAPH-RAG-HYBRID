static int img_resize(int argc, char **argv)

{

    Error *err = NULL;

    int c, ret, relative;

    const char *filename, *fmt, *size;

    int64_t n, total_size;

    bool quiet = false;

    BlockBackend *blk = NULL;

    QemuOpts *param;



    static QemuOptsList resize_options = {

        .name = "resize_options",

        .head = QTAILQ_HEAD_INITIALIZER(resize_options.head),

        .desc = {

            {

                .name = BLOCK_OPT_SIZE,

                .type = QEMU_OPT_SIZE,

                .help = "Virtual disk size"

            }, {

                /* end of list */

            }

        },

    };

    bool image_opts = false;



    /* Remove size from argv manually so that negative numbers are not treated

     * as options by getopt. */

    if (argc < 3) {

        error_exit("Not enough arguments");

        return 1;

    }



    size = argv[--argc];



    /* Parse getopt arguments */

    fmt = NULL;

    for(;;) {

        static const struct option long_options[] = {

            {"help", no_argument, 0, 'h'},

            {"object", required_argument, 0, OPTION_OBJECT},

            {"image-opts", no_argument, 0, OPTION_IMAGE_OPTS},

            {0, 0, 0, 0}

        };

        c = getopt_long(argc, argv, "f:hq",

                        long_options, NULL);

        if (c == -1) {

            break;

        }

        switch(c) {

        case '?':

        case 'h':

            help();

            break;

        case 'f':

            fmt = optarg;

            break;

        case 'q':

            quiet = true;

            break;

        case OPTION_OBJECT: {

            QemuOpts *opts;

            opts = qemu_opts_parse_noisily(&qemu_object_opts,

                                           optarg, true);

            if (!opts) {

                return 1;

            }

        }   break;

        case OPTION_IMAGE_OPTS:

            image_opts = true;

            break;

        }

    }

    if (optind != argc - 1) {

        error_exit("Expecting one image file name");

    }

    filename = argv[optind++];



    if (qemu_opts_foreach(&qemu_object_opts,

                          user_creatable_add_opts_foreach,

                          NULL, NULL)) {

        return 1;

    }



    /* Choose grow, shrink, or absolute resize mode */

    switch (size[0]) {

    case '+':

        relative = 1;

        size++;

        break;

    case '-':

        relative = -1;

        size++;

        break;

    default:

        relative = 0;

        break;

    }



    /* Parse size */

    param = qemu_opts_create(&resize_options, NULL, 0, &error_abort);

    qemu_opt_set(param, BLOCK_OPT_SIZE, size, &err);

    if (err) {

        error_report_err(err);

        ret = -1;

        qemu_opts_del(param);

        goto out;

    }

    n = qemu_opt_get_size(param, BLOCK_OPT_SIZE, 0);

    qemu_opts_del(param);



    blk = img_open(image_opts, filename, fmt,

                   BDRV_O_RDWR | BDRV_O_RESIZE, false, quiet);

    if (!blk) {

        ret = -1;

        goto out;

    }



    if (relative) {

        total_size = blk_getlength(blk) + n * relative;

    } else {

        total_size = n;

    }

    if (total_size <= 0) {

        error_report("New image size must be positive");

        ret = -1;

        goto out;

    }



    ret = blk_truncate(blk, total_size);

    switch (ret) {

    case 0:

        qprintf(quiet, "Image resized.\n");

        break;

    case -ENOTSUP:

        error_report("This image does not support resize");

        break;

    case -EACCES:

        error_report("Image is read-only");

        break;

    default:

        error_report("Error resizing image: %s", strerror(-ret));

        break;

    }

out:

    blk_unref(blk);

    if (ret) {

        return 1;

    }

    return 0;

}
