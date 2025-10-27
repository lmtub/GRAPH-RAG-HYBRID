static int vp3_decode_end(AVCodecContext *avctx)
{
    Vp3DecodeContext *s = avctx->priv_data;
    int i;
    av_free(s->superblock_coding);
    av_free(s->all_fragments);
    av_free(s->coeffs);
    av_free(s->coded_fragment_list);
    av_free(s->superblock_fragments);
    av_free(s->superblock_macroblocks);
    av_free(s->macroblock_fragments);
    av_free(s->macroblock_coding);
    /* release all frames */
    if (s->golden_frame.data[0] && s->golden_frame.data[0] != s->last_frame.data[0])
        avctx->release_buffer(avctx, &s->golden_frame);
    if (s->last_frame.data[0])
        avctx->release_buffer(avctx, &s->last_frame);
    /* no need to release the current_frame since it will always be pointing
     * to the same frame as either the golden or last frame */
    return 0;