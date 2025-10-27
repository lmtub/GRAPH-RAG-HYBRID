static void copy_context_reset(AVCodecContext *avctx)
{
    av_opt_free(avctx);
    av_freep(&avctx->rc_override);
    av_freep(&avctx->intra_matrix);
    av_freep(&avctx->inter_matrix);
    av_freep(&avctx->extradata);
    av_freep(&avctx->subtitle_header);
    av_buffer_unref(&avctx->hw_frames_ctx);
    avctx->subtitle_header_size = 0;
    avctx->extradata_size = 0;
}