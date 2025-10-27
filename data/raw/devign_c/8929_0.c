static int img_resize(int argc, char **argv)

{

    int c, ret, relative;

    const char *filename, *fmt, *size;

    int64_t n, total_size;

    BlockDriverState *bs = NULL;

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



    /* Remove size from argv manually so that negative numbers are not treated

     * as options by getopt. */

    if (argc < 3) {

        help();

        return 1;

    }



    size = argv[--argc];



    /* Parse getopt arguments */

    fmt = NULL;

    for(;;) {

        c = getopt(argc, argv, "f:h");

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

        }

    }

    if (optind >= argc) {

        help();

    }

    filename = argv[optind++];



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

    param = qemu_opts_create(&resize_options, NULL, 0, NULL);

    if (qemu_opt_set(param, BLOCK_OPT_SIZE, size)) {

        /* Error message already printed when size parsing fails */

        ret = -1;

        qemu_opts_del(param);

        goto out;

    }

    n = qemu_opt_get_size(param, BLOCK_OPT_SIZE, 0);

    qemu_opts_del(param);



    bs = bdrv_new_open(filename, fmt, BDRV_O_FLAGS | BDRV_O_RDWR);

    if (!bs) {

        ret = -1;

        goto out;

    }



    if (relative) {

        total_size = bdrv_getlength(bs) + n * relative;

    } else {

        total_size = n;

    }

    if (total_size <= 0) {

        error_report("New image size must be positive");

        ret = -1;

        goto out;

    }



    ret = bdrv_truncate(bs, total_size);

    switch (ret) {

    case 0:

        printf("Image resized.\n");

        break;

    case -ENOTSUP:

        error_report("This image does not support resize");

        break;

    case -EACCES:

        error_report("Image is read-only");

        break;

    default:

        error_report("Error resizing image (%d)", -ret);

        break;

    }

out:

    if (bs) {

        bdrv_delete(bs);

    }

    if (ret) {

        return 1;

    }

    return 0;

}
