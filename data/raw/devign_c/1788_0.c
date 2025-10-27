static av_cold int libx265_encode_close(AVCodecContext *avctx)

{

    libx265Context *ctx = avctx->priv_data;



    av_frame_free(&avctx->coded_frame);



    ctx->api->param_free(ctx->params);



    if (ctx->encoder)

        ctx->api->encoder_close(ctx->encoder);



    return 0;

}
