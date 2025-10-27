static av_cold int vdadec_close(AVCodecContext *avctx)

{

    VDADecoderContext *ctx = avctx->priv_data;

    /* release buffers and decoder */

    ff_vda_destroy_decoder(&ctx->vda_ctx);

    /* close H.264 decoder */

    if (ctx->h264_initialized)

        ff_h264_decoder.close(avctx);

    return 0;

}
