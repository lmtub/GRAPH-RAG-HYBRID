static int readv_f(int argc, char **argv)

{

    struct timeval t1, t2;

    int Cflag = 0, qflag = 0, vflag = 0;

    int c, cnt;

    char *buf;

    int64_t offset;

    /* Some compilers get confused and warn if this is not initialized.  */

    int total = 0;

    int nr_iov;

    QEMUIOVector qiov;

    int pattern = 0;

    int Pflag = 0;



    while ((c = getopt(argc, argv, "CP:qv")) != EOF) {

        switch (c) {

        case 'C':

            Cflag = 1;

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

        case 'v':

            vflag = 1;

            break;

        default:

            return command_usage(&readv_cmd);

        }

    }



    if (optind > argc - 2) {

        return command_usage(&readv_cmd);

    }





    offset = cvtnum(argv[optind]);

    if (offset < 0) {

        printf("non-numeric length argument -- %s\n", argv[optind]);

        return 0;

    }

    optind++;



    if (offset & 0x1ff) {

        printf("offset %" PRId64 " is not sector aligned\n",

               offset);

        return 0;

    }



    nr_iov = argc - optind;

    buf = create_iovec(&qiov, &argv[optind], nr_iov, 0xab);

    if (buf == NULL) {

        return 0;

    }



    gettimeofday(&t1, NULL);

    cnt = do_aio_readv(&qiov, offset, &total);

    gettimeofday(&t2, NULL);



    if (cnt < 0) {

        printf("readv failed: %s\n", strerror(-cnt));

        goto out;

    }



    if (Pflag) {

        void *cmp_buf = g_malloc(qiov.size);

        memset(cmp_buf, pattern, qiov.size);

        if (memcmp(buf, cmp_buf, qiov.size)) {

            printf("Pattern verification failed at offset %"

                   PRId64 ", %zd bytes\n", offset, qiov.size);

        }

        g_free(cmp_buf);

    }



    if (qflag) {

        goto out;

    }



    if (vflag) {

        dump_buffer(buf, offset, qiov.size);

    }



    /* Finally, report back -- -C gives a parsable format */

    t2 = tsub(t2, t1);

    print_report("read", &t2, offset, qiov.size, total, cnt, Cflag);



out:


    qemu_io_free(buf);

    return 0;

}