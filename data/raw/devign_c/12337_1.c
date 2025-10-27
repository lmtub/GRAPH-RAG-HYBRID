static int write_f(BlockBackend *blk, int argc, char **argv)

{

    struct timeval t1, t2;

    bool Cflag = false, qflag = false, bflag = false;

    bool Pflag = false, zflag = false, cflag = false;

    int flags = 0;

    int c, cnt;

    char *buf = NULL;

    int64_t offset;

    int64_t count;

    /* Some compilers get confused and warn if this is not initialized.  */

    int64_t total = 0;

    int pattern = 0xcd;



    while ((c = getopt(argc, argv, "bcCfpP:quz")) != -1) {

        switch (c) {

        case 'b':

            bflag = true;

            break;

        case 'c':

            cflag = true;

            break;

        case 'C':

            Cflag = true;

            break;

        case 'f':

            flags |= BDRV_REQ_FUA;

            break;

        case 'p':

            /* Ignored for backwards compatibility */

            break;

        case 'P':

            Pflag = true;

            pattern = parse_pattern(optarg);

            if (pattern < 0) {

                return 0;

            }

            break;

        case 'q':

            qflag = true;

            break;

        case 'u':

            flags |= BDRV_REQ_MAY_UNMAP;

            break;

        case 'z':

            zflag = true;

            break;

        default:

            return qemuio_command_usage(&write_cmd);

        }

    }



    if (optind != argc - 2) {

        return qemuio_command_usage(&write_cmd);

    }



    if (bflag && zflag) {

        printf("-b and -z cannot be specified at the same time\n");

        return 0;

    }



    if ((flags & BDRV_REQ_FUA) && (bflag || cflag)) {

        printf("-f and -b or -c cannot be specified at the same time\n");

        return 0;

    }



    if ((flags & BDRV_REQ_MAY_UNMAP) && !zflag) {

        printf("-u requires -z to be specified\n");

        return 0;

    }



    if (zflag && Pflag) {

        printf("-z and -P cannot be specified at the same time\n");

        return 0;

    }



    offset = cvtnum(argv[optind]);

    if (offset < 0) {

        print_cvtnum_err(offset, argv[optind]);

        return 0;

    }



    optind++;

    count = cvtnum(argv[optind]);

    if (count < 0) {

        print_cvtnum_err(count, argv[optind]);

        return 0;

    } else if (count > SIZE_MAX) {

        printf("length cannot exceed %" PRIu64 ", given %s\n",

               (uint64_t) SIZE_MAX, argv[optind]);

        return 0;

    }



    if (bflag || cflag) {

        if (offset & 0x1ff) {

            printf("offset %" PRId64 " is not sector aligned\n",

                   offset);

            return 0;

        }



        if (count & 0x1ff) {

            printf("count %"PRId64" is not sector aligned\n",

                   count);

            return 0;

        }

    }



    if (!zflag) {

        buf = qemu_io_alloc(blk, count, pattern);

    }



    gettimeofday(&t1, NULL);

    if (bflag) {

        cnt = do_save_vmstate(blk, buf, offset, count, &total);

    } else if (zflag) {

        cnt = do_co_pwrite_zeroes(blk, offset, count, flags, &total);

    } else if (cflag) {

        cnt = do_write_compressed(blk, buf, offset, count, &total);

    } else {

        cnt = do_pwrite(blk, buf, offset, count, flags, &total);

    }

    gettimeofday(&t2, NULL);



    if (cnt < 0) {

        printf("write failed: %s\n", strerror(-cnt));

        goto out;

    }



    if (qflag) {

        goto out;

    }



    /* Finally, report back -- -C gives a parsable format */

    t2 = tsub(t2, t1);

    print_report("wrote", &t2, offset, count, total, cnt, Cflag);



out:

    if (!zflag) {

        qemu_io_free(buf);

    }



    return 0;

}
