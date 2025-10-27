static int multiwrite_f(BlockBackend *blk, int argc, char **argv)

{

    struct timeval t1, t2;

    int Cflag = 0, qflag = 0;

    int c, cnt;

    char **buf;

    int64_t offset, first_offset = 0;

    /* Some compilers get confused and warn if this is not initialized.  */

    int total = 0;

    int nr_iov;

    int nr_reqs;

    int pattern = 0xcd;

    QEMUIOVector *qiovs;

    int i;

    BlockRequest *reqs;



    while ((c = getopt(argc, argv, "CqP:")) != EOF) {

        switch (c) {

        case 'C':

            Cflag = 1;

            break;

        case 'q':

            qflag = 1;

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



    nr_reqs = 1;

    for (i = optind; i < argc; i++) {

        if (!strcmp(argv[i], ";")) {

            nr_reqs++;

        }

    }



    reqs = g_new0(BlockRequest, nr_reqs);

    buf = g_new0(char *, nr_reqs);

    qiovs = g_new(QEMUIOVector, nr_reqs);



    for (i = 0; i < nr_reqs && optind < argc; i++) {

        int j;



        /* Read the offset of the request */

        offset = cvtnum(argv[optind]);

        if (offset < 0) {

            printf("non-numeric offset argument -- %s\n", argv[optind]);

            goto out;

        }

        optind++;



        if (offset & 0x1ff) {

            printf("offset %lld is not sector aligned\n",

                   (long long)offset);

            goto out;

        }



        if (i == 0) {

            first_offset = offset;

        }



        /* Read lengths for qiov entries */

        for (j = optind; j < argc; j++) {

            if (!strcmp(argv[j], ";")) {

                break;

            }

        }



        nr_iov = j - optind;



        /* Build request */

        buf[i] = create_iovec(blk, &qiovs[i], &argv[optind], nr_iov, pattern);

        if (buf[i] == NULL) {

            goto out;

        }



        reqs[i].qiov = &qiovs[i];

        reqs[i].sector = offset >> 9;

        reqs[i].nb_sectors = reqs[i].qiov->size >> 9;



        optind = j + 1;



        pattern++;

    }



    /* If there were empty requests at the end, ignore them */

    nr_reqs = i;



    gettimeofday(&t1, NULL);

    cnt = do_aio_multiwrite(blk, reqs, nr_reqs, &total);

    gettimeofday(&t2, NULL);



    if (cnt < 0) {

        printf("aio_multiwrite failed: %s\n", strerror(-cnt));

        goto out;

    }



    if (qflag) {

        goto out;

    }



    /* Finally, report back -- -C gives a parsable format */

    t2 = tsub(t2, t1);

    print_report("wrote", &t2, first_offset, total, total, cnt, Cflag);

out:

    for (i = 0; i < nr_reqs; i++) {

        qemu_io_free(buf[i]);

        if (reqs[i].qiov != NULL) {

            qemu_iovec_destroy(&qiovs[i]);

        }

    }

    g_free(buf);

    g_free(reqs);

    g_free(qiovs);

    return 0;

}
