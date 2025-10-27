static int ac3_decode_init(AVCodecContext *avctx)

{

    AC3DecodeContext *ctx = avctx->priv_data;



    ac3_common_init();



    ff_mdct_init(&ctx->imdct_ctx_256, 8, 1);

    ff_mdct_init(&ctx->imdct_ctx_512, 9, 1);

    ctx->samples = av_mallocz(6 * 256 * sizeof (float));

    if (!ctx->samples) {

        av_log(avctx, AV_LOG_ERROR, "Cannot allocate memory for samples\n");

        return -1;

    }

    dither_seed(&ctx->state, 0);



    return 0;

}
