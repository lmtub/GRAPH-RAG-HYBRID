static av_cold int prores_encode_close(AVCodecContext *avctx)

{

    ProresContext* ctx = avctx->priv_data;

    av_freep(&avctx->coded_frame);

    av_free(ctx->fill_y);

    av_free(ctx->fill_u);

    av_free(ctx->fill_v);



    return 0;

}
