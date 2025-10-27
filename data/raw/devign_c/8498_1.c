static int writev_f(BlockBackend *blk, int argc, char **argv)

{

    struct timeval t1, t2;

    bool Cflag = false, qflag = false;

    int flags = 0;

    int c, cnt;

    char *buf;

    int64_t offset;

    /* Some compilers get confused and warn if this is not initialized.  */

    int total = 0;

    int nr_iov;

    int pattern = 0xcd;

    QEMUIOVector qiov;



    while ((c = getopt(argc, argv, "CqP:")) != -1) {

        switch (c) {

        case 'C':

            Cflag = true;

            break;

        case 'f':

            flags |= BDRV_REQ_FUA;

            break;

        case 'q':

            qflag = true;

            break;

        case 'P':

            pattern = parse_pattern(optarg);

            if (pattern < 0) {

                return 0;

            }

            break;

        default:

            return qemuio_command_usage(&writev_cmd);

        }

    }



    if (optind > argc - 2) {

        return qemuio_command_usage(&writev_cmd);

    }



    offset = cvtnum(argv[optind]);

    if (offset < 0) {

        print_cvtnum_err(offset, argv[optind]);

        return 0;

    }

    optind++;



    nr_iov = argc - optind;

    buf = create_iovec(blk, &qiov, &argv[optind], nr_iov, pattern);

    if (buf == NULL) {

        return 0;

    }



    gettimeofday(&t1, NULL);

    cnt = do_aio_writev(blk, &qiov, offset, flags, &total);

    gettimeofday(&t2, NULL);



    if (cnt < 0) {

        printf("writev failed: %s\n", strerror(-cnt));

        goto out;

    }



    if (qflag) {

        goto out;

    }



    /* Finally, report back -- -C gives a parsable format */

    t2 = tsub(t2, t1);

    print_report("wrote", &t2, offset, qiov.size, total, cnt, Cflag);

out:

    qemu_iovec_destroy(&qiov);

    qemu_io_free(buf);

    return 0;

}
