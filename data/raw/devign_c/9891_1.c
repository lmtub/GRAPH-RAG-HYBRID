aio_write_done(void *opaque, int ret)

{

	struct aio_ctx *ctx = opaque;

	struct timeval t2;



	gettimeofday(&t2, NULL);





	if (ret < 0) {

		printf("aio_write failed: %s\n", strerror(-ret));

		return;

	}



	if (ctx->qflag) {

		return;

	}



	/* Finally, report back -- -C gives a parsable format */

	t2 = tsub(t2, ctx->t1);

	print_report("wrote", &t2, ctx->offset, ctx->qiov.size,

		     ctx->qiov.size, 1, ctx->Cflag);



	qemu_io_free(ctx->buf);

	free(ctx);

}
