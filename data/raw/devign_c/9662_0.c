static void test_copy(const AVCodec *c1, const AVCodec *c2)

{

    AVCodecContext *ctx1, *ctx2;

    printf("%s -> %s\nclosed:\n", c1 ? c1->name : "NULL", c2 ? c2->name : "NULL");

    ctx1 = avcodec_alloc_context3(c1);

    ctx2 = avcodec_alloc_context3(c2);

    ctx1->width = ctx1->height = 128;

    if (ctx2->codec && ctx2->codec->priv_class && ctx2->codec->priv_data_size) {

        av_opt_set(ctx2->priv_data, "num", "667", 0);

        av_opt_set(ctx2->priv_data, "str", "i'm dest value before copy", 0);

    }

    avcodec_copy_context(ctx2, ctx1);

    test_copy_print_codec(ctx1);

    test_copy_print_codec(ctx2);

    if (ctx1->codec) {

        printf("opened:\n");

        avcodec_open2(ctx1, ctx1->codec, NULL);

        if (ctx2->codec && ctx2->codec->priv_class && ctx2->codec->priv_data_size) {

            av_opt_set(ctx2->priv_data, "num", "667", 0);

            av_opt_set(ctx2->priv_data, "str", "i'm dest value before copy", 0);

        }

        avcodec_copy_context(ctx2, ctx1);

        test_copy_print_codec(ctx1);

        test_copy_print_codec(ctx2);

        avcodec_close(ctx1);

    }

    avcodec_free_context(&ctx1);

    avcodec_free_context(&ctx2);

}
