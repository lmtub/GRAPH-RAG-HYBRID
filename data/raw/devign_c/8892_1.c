read_f(int argc, char **argv)

{

	struct timeval t1, t2;

	int Cflag = 0, pflag = 0, qflag = 0, vflag = 0;

	int Pflag = 0, sflag = 0, lflag = 0;

	int c, cnt;

	char *buf;

	int64_t offset;

	int count;

        /* Some compilers get confused and warn if this is not initialized.  */

        int total = 0;

	int pattern = 0, pattern_offset = 0, pattern_count = 0;



	while ((c = getopt(argc, argv, "Cl:pP:qs:v")) != EOF) {

		switch (c) {

		case 'C':

			Cflag = 1;

			break;

		case 'l':

			lflag = 1;

			pattern_count = cvtnum(optarg);

			if (pattern_count < 0) {

				printf("non-numeric length argument -- %s\n", optarg);

				return 0;

			}

			break;

		case 'p':

			pflag = 1;

			break;

		case 'P':

			Pflag = 1;

			pattern = atoi(optarg);

			break;

		case 'q':

			qflag = 1;

			break;

		case 's':

			sflag = 1;

			pattern_offset = cvtnum(optarg);

			if (pattern_offset < 0) {

				printf("non-numeric length argument -- %s\n", optarg);

				return 0;

			}

			break;

		case 'v':

			vflag = 1;

			break;

		default:

			return command_usage(&read_cmd);

		}

	}



	if (optind != argc - 2)

		return command_usage(&read_cmd);



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



    if (!Pflag && (lflag || sflag)) {

        return command_usage(&read_cmd);

    }



    if (!lflag) {

        pattern_count = count - pattern_offset;

    }



    if ((pattern_count < 0) || (pattern_count + pattern_offset > count))  {

        printf("pattern verfication range exceeds end of read data\n");

        return 0;

    }



	if (!pflag)

		if (offset & 0x1ff) {

			printf("offset %lld is not sector aligned\n",

				(long long)offset);

			return 0;



		if (count & 0x1ff) {

			printf("count %d is not sector aligned\n",

				count);

			return 0;

		}

	}



	buf = qemu_io_alloc(count, 0xab);



	gettimeofday(&t1, NULL);

	if (pflag)

		cnt = do_pread(buf, offset, count, &total);

	else

		cnt = do_read(buf, offset, count, &total);

	gettimeofday(&t2, NULL);



	if (cnt < 0) {

		printf("read failed: %s\n", strerror(-cnt));

		return 0;

	}



	if (Pflag) {

		void* cmp_buf = malloc(pattern_count);

		memset(cmp_buf, pattern, pattern_count);

		if (memcmp(buf + pattern_offset, cmp_buf, pattern_count)) {

			printf("Pattern verification failed at offset %lld, "

				"%d bytes\n",

				(long long) offset + pattern_offset, pattern_count);

		}

		free(cmp_buf);

	}



	if (qflag)

		return 0;



        if (vflag)

		dump_buffer(buf, offset, count);



	/* Finally, report back -- -C gives a parsable format */

	t2 = tsub(t2, t1);

	print_report("read", &t2, offset, count, total, cnt, Cflag);



	qemu_io_free(buf);



	return 0;

}
