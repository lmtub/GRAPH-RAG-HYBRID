av_cold int ff_vp8_decode_free(AVCodecContext *avctx)
{
    VP8Context *s = avctx->priv_data;
    int i;
    vp8_decode_flush_impl(avctx, 1);
    for (i = 0; i < FF_ARRAY_ELEMS(s->frames); i++)
        av_frame_free(&s->frames[i].tf.f);
}