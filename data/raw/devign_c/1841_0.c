static av_cold int vp8_free(AVCodecContext *avctx)

{

    VP8Context *ctx = avctx->priv_data;



    vpx_codec_destroy(&ctx->encoder);

    av_freep(&ctx->twopass_stats.buf);

    av_freep(&avctx->coded_frame);

    av_freep(&avctx->stats_out);

    free_frame_list(ctx->coded_frame_list);

    return 0;

}
