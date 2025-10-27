static av_cold int indeo3_decode_end(AVCodecContext *avctx)
{
    Indeo3DecodeContext *s = avctx->priv_data;
    iv_free_func(s);
    return 0;
}