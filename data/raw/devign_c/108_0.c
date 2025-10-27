aio_write_f(int argc, char **argv)

{

	char *p;

	int count = 0;

	int nr_iov, i, c;

	int pattern = 0xcd;

	struct aio_ctx *ctx = calloc(1, sizeof(struct aio_ctx));

	BlockDriverAIOCB *acb;



	while ((c = getopt(argc, argv, "CqP:")) != EOF) {

		switch (c) {

		case 'C':

			ctx->Cflag = 1;

			break;

		case 'q':

			ctx->qflag = 1;

			break;

		case 'P':

			pattern = atoi(optarg);

			break;

		default:

			return command_usage(&aio_write_cmd);

		}

	}



	if (optind > argc - 2)

		return command_usage(&aio_write_cmd);



	ctx->offset = cvtnum(argv[optind]);

	if (ctx->offset < 0) {

		printf("non-numeric length argument -- %s\n", argv[optind]);

		return 0;

	}

	optind++;



	if (ctx->offset & 0x1ff) {

		printf("offset %lld is not sector aligned\n",

			(long long)ctx->offset);

		return 0;

	}



	if (count & 0x1ff) {

		printf("count %d is not sector aligned\n",

			count);

		return 0;

	}



	for (i = optind; i < argc; i++) {

	        size_t len;



		len = cvtnum(argv[optind]);

		if (len < 0) {

			printf("non-numeric length argument -- %s\n", argv[i]);

			return 0;

		}

		count += len;

	}



	nr_iov = argc - optind;

	qemu_iovec_init(&ctx->qiov, nr_iov);

	ctx->buf = p = qemu_io_alloc(count, pattern);

	for (i = 0; i < nr_iov; i++) {

	        size_t len;



		len = cvtnum(argv[optind]);

		if (len < 0) {

			printf("non-numeric length argument -- %s\n",

				argv[optind]);

			return 0;

		}



		qemu_iovec_add(&ctx->qiov, p, len);

		p += len;

		optind++;

	}



	gettimeofday(&ctx->t1, NULL);

	acb = bdrv_aio_writev(bs, ctx->offset >> 9, &ctx->qiov,

			      ctx->qiov.size >> 9, aio_write_done, ctx);

	if (!acb)

		return -EIO;



	return 0;

}
