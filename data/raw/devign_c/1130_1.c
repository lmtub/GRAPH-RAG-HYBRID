static int aio_write_f(BlockBackend *blk, int argc, char **argv)

{

    int nr_iov, c;

    int pattern = 0xcd;

    struct aio_ctx *ctx = g_new0(struct aio_ctx, 1);



    ctx->blk = blk;

    while ((c = getopt(argc, argv, "CqP:z")) != -1) {

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

        case 'z':

            ctx->zflag = 1;

            break;

        default:


            return qemuio_command_usage(&aio_write_cmd);

        }

    }



    if (optind > argc - 2) {


        return qemuio_command_usage(&aio_write_cmd);

    }



    if (ctx->zflag && optind != argc - 2) {

        printf("-z supports only a single length parameter\n");


        return 0;

    }



    if (ctx->zflag && ctx->Pflag) {

        printf("-z and -P cannot be specified at the same time\n");


        return 0;

    }



    ctx->offset = cvtnum(argv[optind]);

    if (ctx->offset < 0) {

        print_cvtnum_err(ctx->offset, argv[optind]);


        return 0;

    }

    optind++;



    if (ctx->offset & 0x1ff) {

        printf("offset %" PRId64 " is not sector aligned\n",

               ctx->offset);

        block_acct_invalid(blk_get_stats(blk), BLOCK_ACCT_WRITE);


        return 0;

    }



    if (ctx->zflag) {

        int64_t count = cvtnum(argv[optind]);

        if (count < 0) {

            print_cvtnum_err(count, argv[optind]);


            return 0;

        }



        ctx->qiov.size = count;

        blk_aio_write_zeroes(blk, ctx->offset >> 9, count >> 9, 0,

                             aio_write_done, ctx);

    } else {

        nr_iov = argc - optind;

        ctx->buf = create_iovec(blk, &ctx->qiov, &argv[optind], nr_iov,

                                pattern);

        if (ctx->buf == NULL) {

            block_acct_invalid(blk_get_stats(blk), BLOCK_ACCT_WRITE);


            return 0;

        }



        gettimeofday(&ctx->t1, NULL);

        block_acct_start(blk_get_stats(blk), &ctx->acct, ctx->qiov.size,

                         BLOCK_ACCT_WRITE);



        blk_aio_writev(blk, ctx->offset >> 9, &ctx->qiov,

                       ctx->qiov.size >> 9, aio_write_done, ctx);

    }

    return 0;

}