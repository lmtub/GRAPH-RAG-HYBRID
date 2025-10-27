static int teletext_init_decoder(AVCodecContext *avctx)

{

    TeletextContext *ctx = avctx->priv_data;

    unsigned int maj, min, rev;



    vbi_version(&maj, &min, &rev);

    if (!(maj > 0 || min > 2 || min == 2 && rev >= 26)) {

        av_log(avctx, AV_LOG_ERROR, "decoder needs zvbi version >= 0.2.26.\n");

        return AVERROR_EXTERNAL;

    }



    if (ctx->format_id == 0) {

        avctx->width  = 41 * BITMAP_CHAR_WIDTH;

        avctx->height = 25 * BITMAP_CHAR_HEIGHT;

    }



    ctx->dx = NULL;

    ctx->vbi = NULL;

    ctx->pts = AV_NOPTS_VALUE;



#ifdef DEBUG

    {

        char *t;

        ctx->ex = vbi_export_new("text", &t);

    }

#endif

    av_log(avctx, AV_LOG_VERBOSE, "page filter: %s\n", ctx->pgno);

    return (ctx->format_id == 1) ? ff_ass_subtitle_header_default(avctx) : 0;

}
