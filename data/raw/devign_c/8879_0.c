static int CUDAAPI cuvid_handle_video_sequence(void *opaque, CUVIDEOFORMAT* format)

{

    AVCodecContext *avctx = opaque;

    CuvidContext *ctx = avctx->priv_data;

    AVHWFramesContext *hwframe_ctx = (AVHWFramesContext*)ctx->hwframe->data;

    CUVIDDECODECREATEINFO cuinfo;

    int surface_fmt;



    enum AVPixelFormat pix_fmts[3] = { AV_PIX_FMT_CUDA,

                                       AV_PIX_FMT_NONE,  // Will be updated below

                                       AV_PIX_FMT_NONE };



    av_log(avctx, AV_LOG_TRACE, "pfnSequenceCallback, progressive_sequence=%d\n", format->progressive_sequence);



    ctx->internal_error = 0;



    switch (format->bit_depth_luma_minus8) {

    case 0: // 8-bit

        pix_fmts[1] = AV_PIX_FMT_NV12;

        break;

    case 2: // 10-bit

        pix_fmts[1] = AV_PIX_FMT_P010;

        break;

    case 4: // 12-bit

        pix_fmts[1] = AV_PIX_FMT_P016;

        break;

    default:

        av_log(avctx, AV_LOG_ERROR, "unsupported bit depth: %d\n",

               format->bit_depth_luma_minus8 + 8);

        ctx->internal_error = AVERROR(EINVAL);

        return 0;

    }

    surface_fmt = ff_get_format(avctx, pix_fmts);

    if (surface_fmt < 0) {

        av_log(avctx, AV_LOG_ERROR, "ff_get_format failed: %d\n", surface_fmt);

        ctx->internal_error = AVERROR(EINVAL);

        return 0;

    }



    av_log(avctx, AV_LOG_VERBOSE, "Formats: Original: %s | HW: %s | SW: %s\n",

           av_get_pix_fmt_name(avctx->pix_fmt),

           av_get_pix_fmt_name(surface_fmt),

           av_get_pix_fmt_name(avctx->sw_pix_fmt));



    avctx->pix_fmt = surface_fmt;



    // Update our hwframe ctx, as the get_format callback might have refreshed it!

    if (avctx->hw_frames_ctx) {

        av_buffer_unref(&ctx->hwframe);



        ctx->hwframe = av_buffer_ref(avctx->hw_frames_ctx);

        if (!ctx->hwframe) {

            ctx->internal_error = AVERROR(ENOMEM);

            return 0;

        }



        hwframe_ctx = (AVHWFramesContext*)ctx->hwframe->data;

    }



    avctx->width = format->display_area.right;

    avctx->height = format->display_area.bottom;



    ff_set_sar(avctx, av_div_q(

        (AVRational){ format->display_aspect_ratio.x, format->display_aspect_ratio.y },

        (AVRational){ avctx->width, avctx->height }));



    if (!format->progressive_sequence && ctx->deint_mode == cudaVideoDeinterlaceMode_Weave)

        avctx->flags |= AV_CODEC_FLAG_INTERLACED_DCT;

    else

        avctx->flags &= ~AV_CODEC_FLAG_INTERLACED_DCT;



    if (format->video_signal_description.video_full_range_flag)

        avctx->color_range = AVCOL_RANGE_JPEG;

    else

        avctx->color_range = AVCOL_RANGE_MPEG;



    avctx->color_primaries = format->video_signal_description.color_primaries;

    avctx->color_trc = format->video_signal_description.transfer_characteristics;

    avctx->colorspace = format->video_signal_description.matrix_coefficients;



    if (format->bitrate)

        avctx->bit_rate = format->bitrate;



    if (format->frame_rate.numerator && format->frame_rate.denominator) {

        avctx->framerate.num = format->frame_rate.numerator;

        avctx->framerate.den = format->frame_rate.denominator;

    }



    if (ctx->cudecoder

            && avctx->coded_width == format->coded_width

            && avctx->coded_height == format->coded_height

            && ctx->chroma_format == format->chroma_format

            && ctx->codec_type == format->codec)

        return 1;



    if (ctx->cudecoder) {

        av_log(avctx, AV_LOG_TRACE, "Re-initializing decoder\n");

        ctx->internal_error = CHECK_CU(ctx->cvdl->cuvidDestroyDecoder(ctx->cudecoder));

        if (ctx->internal_error < 0)

            return 0;

        ctx->cudecoder = NULL;

    }



    if (hwframe_ctx->pool && (

            hwframe_ctx->width < avctx->width ||

            hwframe_ctx->height < avctx->height ||

            hwframe_ctx->format != AV_PIX_FMT_CUDA ||

            hwframe_ctx->sw_format != avctx->sw_pix_fmt)) {

        av_log(avctx, AV_LOG_ERROR, "AVHWFramesContext is already initialized with incompatible parameters\n");

        ctx->internal_error = AVERROR(EINVAL);

        return 0;

    }



    if (format->chroma_format != cudaVideoChromaFormat_420) {

        av_log(avctx, AV_LOG_ERROR, "Chroma formats other than 420 are not supported\n");

        ctx->internal_error = AVERROR(EINVAL);

        return 0;

    }



    avctx->coded_width = format->coded_width;

    avctx->coded_height = format->coded_height;



    ctx->chroma_format = format->chroma_format;



    memset(&cuinfo, 0, sizeof(cuinfo));



    cuinfo.CodecType = ctx->codec_type = format->codec;

    cuinfo.ChromaFormat = format->chroma_format;



    switch (avctx->sw_pix_fmt) {

    case AV_PIX_FMT_NV12:

        cuinfo.OutputFormat = cudaVideoSurfaceFormat_NV12;

        break;

    case AV_PIX_FMT_P010:

    case AV_PIX_FMT_P016:

        cuinfo.OutputFormat = cudaVideoSurfaceFormat_P016;

        break;

    default:

        av_log(avctx, AV_LOG_ERROR, "Output formats other than NV12, P010 or P016 are not supported\n");

        ctx->internal_error = AVERROR(EINVAL);

        return 0;

    }



    cuinfo.ulWidth = avctx->coded_width;

    cuinfo.ulHeight = avctx->coded_height;

    cuinfo.ulTargetWidth = cuinfo.ulWidth;

    cuinfo.ulTargetHeight = cuinfo.ulHeight;



    cuinfo.target_rect.left = 0;

    cuinfo.target_rect.top = 0;

    cuinfo.target_rect.right = cuinfo.ulWidth;

    cuinfo.target_rect.bottom = cuinfo.ulHeight;



    cuinfo.ulNumDecodeSurfaces = ctx->nb_surfaces;

    cuinfo.ulNumOutputSurfaces = 1;

    cuinfo.ulCreationFlags = cudaVideoCreate_PreferCUVID;

    cuinfo.bitDepthMinus8 = format->bit_depth_luma_minus8;



    if (format->progressive_sequence) {

        ctx->deint_mode = cuinfo.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;

    } else {

        cuinfo.DeinterlaceMode = ctx->deint_mode;

    }



    if (ctx->deint_mode != cudaVideoDeinterlaceMode_Weave)

        avctx->framerate = av_mul_q(avctx->framerate, (AVRational){2, 1});



    ctx->internal_error = CHECK_CU(ctx->cvdl->cuvidCreateDecoder(&ctx->cudecoder, &cuinfo));

    if (ctx->internal_error < 0)

        return 0;



    if (!hwframe_ctx->pool) {

        hwframe_ctx->format = AV_PIX_FMT_CUDA;

        hwframe_ctx->sw_format = avctx->sw_pix_fmt;

        hwframe_ctx->width = avctx->width;

        hwframe_ctx->height = avctx->height;



        if ((ctx->internal_error = av_hwframe_ctx_init(ctx->hwframe)) < 0) {

            av_log(avctx, AV_LOG_ERROR, "av_hwframe_ctx_init failed\n");

            return 0;

        }

    }



    return 1;

}
