int ff_mediacodec_dec_close(AVCodecContext *avctx, MediaCodecDecContext *s)
{
    if (s->codec) {
        ff_AMediaCodec_delete(s->codec);
        s->codec = NULL;
    }
    if (s->format) {
        ff_AMediaFormat_delete(s->format);
        s->format = NULL;
    }
    return 0;
}