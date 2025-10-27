static int teletext_close_decoder(AVCodecContext *avctx)

{

    TeletextContext *ctx = avctx->priv_data;



    av_dlog(avctx, "lines_total=%u\n", ctx->lines_processed);

    while (ctx->nb_pages)

        subtitle_rect_free(&ctx->pages[--ctx->nb_pages].sub_rect);

    av_freep(&ctx->pages);



    vbi_dvb_demux_delete(ctx->dx);

    vbi_decoder_delete(ctx->vbi);

    ctx->dx = NULL;

    ctx->vbi = NULL;

    ctx->pts = AV_NOPTS_VALUE;

    return 0;

}
