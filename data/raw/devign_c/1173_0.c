static int aio_read_f(int argc, char **argv)

{

    int nr_iov, c;

    struct aio_ctx *ctx = calloc(1, sizeof(struct aio_ctx));



    while ((c = getopt(argc, argv, "CP:qv")) != EOF) {

        switch (c) {

        case 'C':

            ctx->Cflag = 1;

            break;

        case 'P':

            ctx->Pflag = 1;

            ctx->pattern = parse_pattern(optarg);

            if (ctx->pattern < 0) {

                free(ctx);

                return 0;

            }

            break;

        case 'q':

            ctx->qflag = 1;

            break;

        case 'v':

            ctx->vflag = 1;

            break;

        default:

            free(ctx);

            return command_usage(&aio_read_cmd);

        }

    }



    if (optind > argc - 2) {

        free(ctx);

        return command_usage(&aio_read_cmd);

    }



    ctx->offset = cvtnum(argv[optind]);

    if (ctx->offset < 0) {

        printf("non-numeric length argument -- %s\n", argv[optind]);

        free(ctx);

        return 0;

    }

    optind++;



    if (ctx->offset & 0x1ff) {

        printf("offset %" PRId64 " is not sector aligned\n",

               ctx->offset);

        free(ctx);

        return 0;

    }



    nr_iov = argc - optind;

    ctx->buf = create_iovec(&ctx->qiov, &argv[optind], nr_iov, 0xab);

    if (ctx->buf == NULL) {

        free(ctx);

        return 0;

    }



    gettimeofday(&ctx->t1, NULL);

    bdrv_aio_readv(bs, ctx->offset >> 9, &ctx->qiov,

                   ctx->qiov.size >> 9, aio_read_done, ctx);

    return 0;

}
