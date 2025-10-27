static int img_amend(int argc, char **argv)

{

    Error *err = NULL;

    int c, ret = 0;

    char *options = NULL;

    QemuOptsList *create_opts = NULL;

    QemuOpts *opts = NULL;

    const char *fmt = NULL, *filename, *cache;

    int flags;

    bool quiet = false, progress = false;

    BlockBackend *blk = NULL;

    BlockDriverState *bs = NULL;



    cache = BDRV_DEFAULT_CACHE;

    for (;;) {

        c = getopt(argc, argv, "ho:f:t:pq");

        if (c == -1) {

            break;

        }



        switch (c) {

            case 'h':

            case '?':

                help();

                break;

            case 'o':

                if (!is_valid_option_list(optarg)) {

                    error_report("Invalid option list: %s", optarg);

                    ret = -1;

                    goto out;

                }

                if (!options) {

                    options = g_strdup(optarg);

                } else {

                    char *old_options = options;

                    options = g_strdup_printf("%s,%s", options, optarg);

                    g_free(old_options);

                }

                break;

            case 'f':

                fmt = optarg;

                break;

            case 't':

                cache = optarg;

                break;

            case 'p':

                progress = true;

                break;

            case 'q':

                quiet = true;

                break;

        }

    }



    if (!options) {

        error_exit("Must specify options (-o)");

    }



    if (quiet) {

        progress = false;

    }

    qemu_progress_init(progress, 1.0);



    filename = (optind == argc - 1) ? argv[argc - 1] : NULL;

    if (fmt && has_help_option(options)) {

        /* If a format is explicitly specified (and possibly no filename is

         * given), print option help here */

        ret = print_block_option_help(filename, fmt);

        goto out;

    }



    if (optind != argc - 1) {

        error_report("Expecting one image file name");

        ret = -1;

        goto out;

    }



    flags = BDRV_O_FLAGS | BDRV_O_RDWR;

    ret = bdrv_parse_cache_flags(cache, &flags);

    if (ret < 0) {

        error_report("Invalid cache option: %s", cache);

        goto out;

    }



    blk = img_open("image", filename, fmt, flags, true, quiet);

    if (!blk) {

        ret = -1;

        goto out;

    }

    bs = blk_bs(blk);



    fmt = bs->drv->format_name;



    if (has_help_option(options)) {

        /* If the format was auto-detected, print option help here */

        ret = print_block_option_help(filename, fmt);

        goto out;

    }



    if (!bs->drv->create_opts) {

        error_report("Format driver '%s' does not support any options to amend",

                     fmt);

        ret = -1;

        goto out;

    }



    create_opts = qemu_opts_append(create_opts, bs->drv->create_opts);

    opts = qemu_opts_create(create_opts, NULL, 0, &error_abort);

    if (options) {

        qemu_opts_do_parse(opts, options, NULL, &err);

        if (err) {

            error_report_err(err);

            ret = -1;

            goto out;

        }

    }



    /* In case the driver does not call amend_status_cb() */

    qemu_progress_print(0.f, 0);

    ret = bdrv_amend_options(bs, opts, &amend_status_cb);

    qemu_progress_print(100.f, 0);

    if (ret < 0) {

        error_report("Error while amending options: %s", strerror(-ret));

        goto out;

    }



out:

    qemu_progress_end();



    blk_unref(blk);

    qemu_opts_del(opts);

    qemu_opts_free(create_opts);

    g_free(options);



    if (ret) {

        return 1;

    }

    return 0;

}
