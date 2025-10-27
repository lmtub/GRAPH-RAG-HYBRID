static int aio_write_f(int argc, char **argv)

{

    int nr_iov, c;

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

            pattern = parse_pattern(optarg);

            if (pattern < 0) {


                return 0;

            }

            break;

        default:


            return command_usage(&aio_write_cmd);

        }

    }



    if (optind > argc - 2) {


        return command_usage(&aio_write_cmd);

    }



    ctx->offset = cvtnum(argv[optind]);

    if (ctx->offset < 0) {

        printf("non-numeric length argument -- %s\n", argv[optind]);


        return 0;

    }

    optind++;



    if (ctx->offset & 0x1ff) {

        printf("offset %" PRId64 " is not sector aligned\n",

               ctx->offset);


        return 0;

    }



    nr_iov = argc - optind;

    ctx->buf = create_iovec(&ctx->qiov, &argv[optind], nr_iov, pattern);



    gettimeofday(&ctx->t1, NULL);

    acb = bdrv_aio_writev(bs, ctx->offset >> 9, &ctx->qiov,

                          ctx->qiov.size >> 9, aio_write_done, ctx);

    if (!acb) {

        free(ctx->buf);


        return -EIO;

    }



    return 0;

}