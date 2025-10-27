static int img_bench(int argc, char **argv)

{

    int c, ret = 0;

    const char *fmt = NULL, *filename;

    bool quiet = false;

    bool image_opts = false;

    bool is_write = false;

    int count = 75000;

    int depth = 64;

    int64_t offset = 0;

    size_t bufsize = 4096;

    int pattern = 0;

    size_t step = 0;

    int flush_interval = 0;

    bool drain_on_flush = true;

    int64_t image_size;

    BlockBackend *blk = NULL;

    BenchData data = {};

    int flags = 0;

    bool writethrough = false;

    struct timeval t1, t2;

    int i;



    for (;;) {

        static const struct option long_options[] = {

            {"help", no_argument, 0, 'h'},

            {"flush-interval", required_argument, 0, OPTION_FLUSH_INTERVAL},

            {"image-opts", no_argument, 0, OPTION_IMAGE_OPTS},

            {"pattern", required_argument, 0, OPTION_PATTERN},

            {"no-drain", no_argument, 0, OPTION_NO_DRAIN},

            {0, 0, 0, 0}

        };

        c = getopt_long(argc, argv, "hc:d:f:no:qs:S:t:w", long_options, NULL);

        if (c == -1) {

            break;

        }



        switch (c) {

        case 'h':

        case '?':

            help();

            break;

        case 'c':

        {

            unsigned long res;



            if (qemu_strtoul(optarg, NULL, 0, &res) < 0 || res > INT_MAX) {

                error_report("Invalid request count specified");

                return 1;

            }

            count = res;

            break;

        }

        case 'd':

        {

            unsigned long res;



            if (qemu_strtoul(optarg, NULL, 0, &res) < 0 || res > INT_MAX) {

                error_report("Invalid queue depth specified");

                return 1;

            }

            depth = res;

            break;

        }

        case 'f':

            fmt = optarg;

            break;

        case 'n':

            flags |= BDRV_O_NATIVE_AIO;

            break;

        case 'o':

        {

            offset = cvtnum(optarg);

            if (offset < 0) {

                error_report("Invalid offset specified");

                return 1;

            }

            break;

        }

            break;

        case 'q':

            quiet = true;

            break;

        case 's':

        {

            int64_t sval;



            sval = cvtnum(optarg);

            if (sval < 0 || sval > INT_MAX) {

                error_report("Invalid buffer size specified");

                return 1;

            }



            bufsize = sval;

            break;

        }

        case 'S':

        {

            int64_t sval;



            sval = cvtnum(optarg);

            if (sval < 0 || sval > INT_MAX) {

                error_report("Invalid step size specified");

                return 1;

            }



            step = sval;

            break;

        }

        case 't':

            ret = bdrv_parse_cache_mode(optarg, &flags, &writethrough);

            if (ret < 0) {

                error_report("Invalid cache mode");

                ret = -1;

                goto out;

            }

            break;

        case 'w':

            flags |= BDRV_O_RDWR;

            is_write = true;

            break;

        case OPTION_PATTERN:

        {

            unsigned long res;



            if (qemu_strtoul(optarg, NULL, 0, &res) < 0 || res > 0xff) {

                error_report("Invalid pattern byte specified");

                return 1;

            }

            pattern = res;

            break;

        }

        case OPTION_FLUSH_INTERVAL:

        {

            unsigned long res;



            if (qemu_strtoul(optarg, NULL, 0, &res) < 0 || res > INT_MAX) {

                error_report("Invalid flush interval specified");

                return 1;

            }

            flush_interval = res;

            break;

        }

        case OPTION_NO_DRAIN:

            drain_on_flush = false;

            break;

        case OPTION_IMAGE_OPTS:

            image_opts = true;

            break;

        }

    }



    if (optind != argc - 1) {

        error_exit("Expecting one image file name");

    }

    filename = argv[argc - 1];



    if (!is_write && flush_interval) {

        error_report("--flush-interval is only available in write tests");

        ret = -1;

        goto out;

    }

    if (flush_interval && flush_interval < depth) {

        error_report("Flush interval can't be smaller than depth");

        ret = -1;

        goto out;

    }



    blk = img_open(image_opts, filename, fmt, flags, writethrough, quiet);

    if (!blk) {

        ret = -1;

        goto out;

    }



    image_size = blk_getlength(blk);

    if (image_size < 0) {

        ret = image_size;

        goto out;

    }



    data = (BenchData) {

        .blk            = blk,

        .image_size     = image_size,

        .bufsize        = bufsize,

        .step           = step ?: bufsize,

        .nrreq          = depth,

        .n              = count,

        .offset         = offset,

        .write          = is_write,

        .flush_interval = flush_interval,

        .drain_on_flush = drain_on_flush,

    };

    printf("Sending %d %s requests, %d bytes each, %d in parallel "

           "(starting at offset %" PRId64 ", step size %d)\n",

           data.n, data.write ? "write" : "read", data.bufsize, data.nrreq,

           data.offset, data.step);

    if (flush_interval) {

        printf("Sending flush every %d requests\n", flush_interval);

    }



    data.buf = blk_blockalign(blk, data.nrreq * data.bufsize);

    memset(data.buf, pattern, data.nrreq * data.bufsize);



    data.qiov = g_new(QEMUIOVector, data.nrreq);

    for (i = 0; i < data.nrreq; i++) {

        qemu_iovec_init(&data.qiov[i], 1);

        qemu_iovec_add(&data.qiov[i],

                       data.buf + i * data.bufsize, data.bufsize);

    }



    gettimeofday(&t1, NULL);

    bench_cb(&data, 0);



    while (data.n > 0) {

        main_loop_wait(false);

    }

    gettimeofday(&t2, NULL);



    printf("Run completed in %3.3f seconds.\n",

           (t2.tv_sec - t1.tv_sec)

           + ((double)(t2.tv_usec - t1.tv_usec) / 1000000));



out:

    qemu_vfree(data.buf);

    blk_unref(blk);



    if (ret) {

        return 1;

    }

    return 0;

}
