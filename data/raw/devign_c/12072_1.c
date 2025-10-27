static int img_commit(int argc, char **argv)

{

    int c, ret, flags;

    const char *filename, *fmt, *cache, *base;

    BlockBackend *blk;

    BlockDriverState *bs, *base_bs;

    BlockJob *job;

    bool progress = false, quiet = false, drop = false;

    bool writethrough;

    Error *local_err = NULL;

    CommonBlockJobCBInfo cbi;

    bool image_opts = false;

    AioContext *aio_context;



    fmt = NULL;

    cache = BDRV_DEFAULT_CACHE;

    base = NULL;

    for(;;) {

        static const struct option long_options[] = {

            {"help", no_argument, 0, 'h'},

            {"object", required_argument, 0, OPTION_OBJECT},

            {"image-opts", no_argument, 0, OPTION_IMAGE_OPTS},

            {0, 0, 0, 0}

        };

        c = getopt_long(argc, argv, "f:ht:b:dpq",

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

        case 't':

            cache = optarg;

            break;

        case 'b':

            base = optarg;

            /* -b implies -d */

            drop = true;

            break;

        case 'd':

            drop = true;

            break;

        case 'p':

            progress = true;

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



    /* Progress is not shown in Quiet mode */

    if (quiet) {

        progress = false;

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



    flags = BDRV_O_RDWR | BDRV_O_UNMAP;

    ret = bdrv_parse_cache_mode(cache, &flags, &writethrough);

    if (ret < 0) {

        error_report("Invalid cache option: %s", cache);

        return 1;

    }



    blk = img_open(image_opts, filename, fmt, flags, writethrough, quiet);

    if (!blk) {

        return 1;

    }

    bs = blk_bs(blk);



    qemu_progress_init(progress, 1.f);

    qemu_progress_print(0.f, 100);



    if (base) {

        base_bs = bdrv_find_backing_image(bs, base);

        if (!base_bs) {

            error_setg(&local_err,

                       "Did not find '%s' in the backing chain of '%s'",

                       base, filename);

            goto done;

        }

    } else {

        /* This is different from QMP, which by default uses the deepest file in

         * the backing chain (i.e., the very base); however, the traditional

         * behavior of qemu-img commit is using the immediate backing file. */

        base_bs = backing_bs(bs);

        if (!base_bs) {

            error_setg(&local_err, "Image does not have a backing file");

            goto done;

        }

    }



    cbi = (CommonBlockJobCBInfo){

        .errp = &local_err,

        .bs   = bs,

    };



    aio_context = bdrv_get_aio_context(bs);

    aio_context_acquire(aio_context);

    commit_active_start("commit", bs, base_bs, BLOCK_JOB_DEFAULT, 0,

                        BLOCKDEV_ON_ERROR_REPORT, NULL, common_block_job_cb,

                        &cbi, &local_err, false);

    aio_context_release(aio_context);

    if (local_err) {

        goto done;

    }



    /* When the block job completes, the BlockBackend reference will point to

     * the old backing file. In order to avoid that the top image is already

     * deleted, so we can still empty it afterwards, increment the reference

     * counter here preemptively. */

    if (!drop) {

        bdrv_ref(bs);

    }



    job = block_job_get("commit");

    run_block_job(job, &local_err);

    if (local_err) {

        goto unref_backing;

    }



    if (!drop && bs->drv->bdrv_make_empty) {

        ret = bs->drv->bdrv_make_empty(bs);

        if (ret) {

            error_setg_errno(&local_err, -ret, "Could not empty %s",

                             filename);

            goto unref_backing;

        }

    }



unref_backing:

    if (!drop) {

        bdrv_unref(bs);

    }



done:

    qemu_progress_end();



    blk_unref(blk);



    if (local_err) {

        error_report_err(local_err);

        return 1;

    }



    qprintf(quiet, "Image committed.\n");

    return 0;

}
