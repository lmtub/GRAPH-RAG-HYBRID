static int read_f(BlockBackend *blk, int argc, char **argv)

{

    struct timeval t1, t2;

    bool Cflag = false, qflag = false, vflag = false;

    bool Pflag = false, sflag = false, lflag = false, bflag = false;

    int c, cnt;

    char *buf;

    int64_t offset;

    int64_t count;

    /* Some compilers get confused and warn if this is not initialized.  */

    int64_t total = 0;

    int pattern = 0;

    int64_t pattern_offset = 0, pattern_count = 0;



    while ((c = getopt(argc, argv, "bCl:pP:qs:v")) != -1) {

        switch (c) {

        case 'b':

            bflag = true;

            break;

        case 'C':

            Cflag = true;

            break;

        case 'l':

            lflag = true;

            pattern_count = cvtnum(optarg);

            if (pattern_count < 0) {

                print_cvtnum_err(pattern_count, optarg);

                return 0;

            }

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

        case 's':

            sflag = true;

            pattern_offset = cvtnum(optarg);

            if (pattern_offset < 0) {

                print_cvtnum_err(pattern_offset, optarg);

                return 0;

            }

            break;

        case 'v':

            vflag = true;

            break;

        default:

            return qemuio_command_usage(&read_cmd);

        }

    }



    if (optind != argc - 2) {

        return qemuio_command_usage(&read_cmd);

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



    if (!Pflag && (lflag || sflag)) {

        return qemuio_command_usage(&read_cmd);

    }



    if (!lflag) {

        pattern_count = count - pattern_offset;

    }



    if ((pattern_count < 0) || (pattern_count + pattern_offset > count))  {

        printf("pattern verification range exceeds end of read data\n");

        return 0;

    }



    if (bflag) {

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



    buf = qemu_io_alloc(blk, count, 0xab);



    gettimeofday(&t1, NULL);

    if (bflag) {

        cnt = do_load_vmstate(blk, buf, offset, count, &total);

    } else {

        cnt = do_pread(blk, buf, offset, count, &total);

    }

    gettimeofday(&t2, NULL);



    if (cnt < 0) {

        printf("read failed: %s\n", strerror(-cnt));

        goto out;

    }



    if (Pflag) {

        void *cmp_buf = g_malloc(pattern_count);

        memset(cmp_buf, pattern, pattern_count);

        if (memcmp(buf + pattern_offset, cmp_buf, pattern_count)) {

            printf("Pattern verification failed at offset %"

                   PRId64 ", %"PRId64" bytes\n",

                   offset + pattern_offset, pattern_count);

        }

        g_free(cmp_buf);

    }



    if (qflag) {

        goto out;

    }



    if (vflag) {

        dump_buffer(buf, offset, count);

    }



    /* Finally, report back -- -C gives a parsable format */

    t2 = tsub(t2, t1);

    print_report("read", &t2, offset, count, total, cnt, Cflag);



out:

    qemu_io_free(buf);



    return 0;

}
