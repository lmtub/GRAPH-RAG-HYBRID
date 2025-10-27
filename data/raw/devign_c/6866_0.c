static int write_f(BlockBackend *blk, int argc, char **argv)

{

    struct timeval t1, t2;

    int Cflag = 0, pflag = 0, qflag = 0, bflag = 0, Pflag = 0, zflag = 0;

    int cflag = 0;

    int c, cnt;

    char *buf = NULL;

    int64_t offset;

    int count;

    /* Some compilers get confused and warn if this is not initialized.  */

    int total = 0;

    int pattern = 0xcd;



    while ((c = getopt(argc, argv, "bcCpP:qz")) != EOF) {

        switch (c) {

        case 'b':

            bflag = 1;

            break;

        case 'c':

            cflag = 1;

            break;

        case 'C':

            Cflag = 1;

            break;

        case 'p':

            pflag = 1;

            break;

        case 'P':

            Pflag = 1;

            pattern = parse_pattern(optarg);

            if (pattern < 0) {

                return 0;

            }

            break;

        case 'q':

            qflag = 1;

            break;

        case 'z':

            zflag = 1;

            break;

        default:

            return qemuio_command_usage(&write_cmd);

        }

    }



    if (optind != argc - 2) {

        return qemuio_command_usage(&write_cmd);

    }



    if (bflag + pflag + zflag > 1) {

        printf("-b, -p, or -z cannot be specified at the same time\n");

        return 0;

    }



    if (zflag && Pflag) {

        printf("-z and -P cannot be specified at the same time\n");

        return 0;

    }



    offset = cvtnum(argv[optind]);

    if (offset < 0) {

        printf("non-numeric length argument -- %s\n", argv[optind]);

        return 0;

    }



    optind++;

    count = cvtnum(argv[optind]);

    if (count < 0) {

        printf("non-numeric length argument -- %s\n", argv[optind]);

        return 0;

    }



    if (!pflag) {

        if (offset & 0x1ff) {

            printf("offset %" PRId64 " is not sector aligned\n",

                   offset);

            return 0;

        }



        if (count & 0x1ff) {

            printf("count %d is not sector aligned\n",

                   count);

            return 0;

        }

    }



    if (!zflag) {

        buf = qemu_io_alloc(blk, count, pattern);

    }



    gettimeofday(&t1, NULL);

    if (pflag) {

        cnt = do_pwrite(blk, buf, offset, count, &total);

    } else if (bflag) {

        cnt = do_save_vmstate(blk, buf, offset, count, &total);

    } else if (zflag) {

        cnt = do_co_write_zeroes(blk, offset, count, &total);

    } else if (cflag) {

        cnt = do_write_compressed(blk, buf, offset, count, &total);

    } else {

        cnt = do_write(blk, buf, offset, count, &total);

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
