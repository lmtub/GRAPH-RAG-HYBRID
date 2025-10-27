static av_cold int dfa_decode_init(AVCodecContext *avctx)
{
    DfaContext *s = avctx->priv_data;
    avctx->pix_fmt = PIX_FMT_PAL8;
    s->frame_buf = av_mallocz(avctx->width * avctx->height + AV_LZO_OUTPUT_PADDING);
    if (!s->frame_buf)
        return AVERROR(ENOMEM);
    return 0;
}