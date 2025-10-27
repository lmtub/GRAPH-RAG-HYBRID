static av_cold int decode_close_mp3on4(AVCodecContext * avctx)
{
    MP3On4DecodeContext *s = avctx->priv_data;
    int i;
    for (i = 0; i < s->frames; i++)
        av_freep(&s->mp3decctx[i]);
    return 0;
}