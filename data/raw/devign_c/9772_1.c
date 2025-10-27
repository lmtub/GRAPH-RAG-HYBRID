static int vda_h264_end_frame(AVCodecContext *avctx)
{
    H264Context *h                      = avctx->priv_data;
    struct vda_context *vda_ctx         = avctx->hwaccel_context;
    AVFrame *frame                      = &h->cur_pic_ptr->f;
    struct vda_buffer *context;
    AVBufferRef *buffer;
    int status;
    if (!vda_ctx->decoder || !vda_ctx->priv_bitstream)
    status = vda_sync_decode(vda_ctx);
    frame->data[3] = (void*)vda_ctx->cv_buffer;
    if (status)
        av_log(avctx, AV_LOG_ERROR, "Failed to decode frame (%d)\n", status);