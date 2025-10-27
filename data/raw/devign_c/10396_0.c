int ff_vdpau_common_end_frame(AVCodecContext *avctx, AVFrame *frame,

                              struct vdpau_picture_context *pic_ctx)

{

    VDPAUContext *vdctx = avctx->internal->hwaccel_priv_data;

    AVVDPAUContext *hwctx = avctx->hwaccel_context;

    VdpVideoSurface surf = ff_vdpau_get_surface_id(frame);

    VdpStatus status;

    int val;



    val = ff_vdpau_common_reinit(avctx);

    if (val < 0)

        return val;



#if FF_API_BUFS_VDPAU

FF_DISABLE_DEPRECATION_WARNINGS

    hwctx->info = pic_ctx->info;

    hwctx->bitstream_buffers = pic_ctx->bitstream_buffers;

    hwctx->bitstream_buffers_used = pic_ctx->bitstream_buffers_used;

    hwctx->bitstream_buffers_allocated = pic_ctx->bitstream_buffers_allocated;

FF_ENABLE_DEPRECATION_WARNINGS

#endif



    if (!hwctx->render) {

        status = hwctx->render2(avctx, frame, (void *)&pic_ctx->info,

                                pic_ctx->bitstream_buffers_used, pic_ctx->bitstream_buffers);

    } else

    status = vdctx->render(vdctx->decoder, surf, (void *)&pic_ctx->info,

                           pic_ctx->bitstream_buffers_used,

                           pic_ctx->bitstream_buffers);



    av_freep(&pic_ctx->bitstream_buffers);



#if FF_API_BUFS_VDPAU

FF_DISABLE_DEPRECATION_WARNINGS

    hwctx->bitstream_buffers = NULL;

    hwctx->bitstream_buffers_used = 0;

    hwctx->bitstream_buffers_allocated = 0;

FF_ENABLE_DEPRECATION_WARNINGS

#endif



    return vdpau_error(status);

}
