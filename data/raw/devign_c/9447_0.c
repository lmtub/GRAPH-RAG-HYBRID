int ff_nvdec_frame_params(AVCodecContext *avctx,

                          AVBufferRef *hw_frames_ctx,

                          int dpb_size)

{

    AVHWFramesContext *frames_ctx = (AVHWFramesContext*)hw_frames_ctx->data;

    const AVPixFmtDescriptor *sw_desc;

    int cuvid_codec_type, cuvid_chroma_format;



    sw_desc = av_pix_fmt_desc_get(avctx->sw_pix_fmt);

    if (!sw_desc)

        return AVERROR_BUG;



    cuvid_codec_type = map_avcodec_id(avctx->codec_id);

    if (cuvid_codec_type < 0) {

        av_log(avctx, AV_LOG_ERROR, "Unsupported codec ID\n");

        return AVERROR_BUG;

    }



    cuvid_chroma_format = map_chroma_format(avctx->sw_pix_fmt);

    if (cuvid_chroma_format < 0) {

        av_log(avctx, AV_LOG_VERBOSE, "Unsupported chroma format\n");

        return AVERROR(EINVAL);

    }



    if (avctx->thread_type & FF_THREAD_FRAME)

        dpb_size += avctx->thread_count;



    frames_ctx->format            = AV_PIX_FMT_CUDA;

    frames_ctx->width             = avctx->coded_width;

    frames_ctx->height            = avctx->coded_height;

    frames_ctx->sw_format         = sw_desc->comp[0].depth > 8 ?

                                    AV_PIX_FMT_P010 : AV_PIX_FMT_NV12;

    frames_ctx->initial_pool_size = dpb_size;



    return 0;

}
