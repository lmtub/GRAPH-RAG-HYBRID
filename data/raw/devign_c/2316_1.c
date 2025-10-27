static int CUDAAPI cuvid_handle_video_sequence(void *opaque, CUVIDEOFORMAT* format)
{
    AVCodecContext *avctx = opaque;
    CuvidContext *ctx = avctx->priv_data;
    AVHWFramesContext *hwframe_ctx = (AVHWFramesContext*)ctx->hwframe->data;
    CUVIDDECODECREATEINFO cuinfo;
    av_log(avctx, AV_LOG_TRACE, "pfnSequenceCallback\n");
    ctx->internal_error = 0;
    avctx->width = format->display_area.right;
    avctx->height = format->display_area.bottom;
    ff_set_sar(avctx, av_div_q(
        (AVRational){ format->display_aspect_ratio.x, format->display_aspect_ratio.y },
        (AVRational){ avctx->width, avctx->height }));
    if (!format->progressive_sequence)
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
    if (ctx->cudecoder
            && avctx->coded_width == format->coded_width
            && avctx->coded_height == format->coded_height
            && ctx->chroma_format == format->chroma_format
            && ctx->codec_type == format->codec)
        return 1;
    if (ctx->cudecoder) {
        av_log(avctx, AV_LOG_ERROR, "re-initializing decoder is not supported\n");
    if (hwframe_ctx->pool && !ctx->ever_flushed) {
        av_log(avctx, AV_LOG_ERROR, "AVHWFramesContext is already initialized\n");
    avctx->coded_width = format->coded_width;
    avctx->coded_height = format->coded_height;
    ctx->chroma_format = format->chroma_format;
    memset(&cuinfo, 0, sizeof(cuinfo));
    cuinfo.CodecType = ctx->codec_type = format->codec;
    cuinfo.ChromaFormat = format->chroma_format;
    cuinfo.OutputFormat = cudaVideoSurfaceFormat_NV12;
    cuinfo.ulWidth = avctx->coded_width;
    cuinfo.ulHeight = avctx->coded_height;
    cuinfo.ulTargetWidth = cuinfo.ulWidth;
    cuinfo.ulTargetHeight = cuinfo.ulHeight;
    cuinfo.target_rect.left = 0;
    cuinfo.target_rect.top = 0;
    cuinfo.target_rect.right = cuinfo.ulWidth;
    cuinfo.target_rect.bottom = cuinfo.ulHeight;
    cuinfo.ulNumDecodeSurfaces = MAX_FRAME_COUNT;
    cuinfo.ulNumOutputSurfaces = 1;
    cuinfo.ulCreationFlags = cudaVideoCreate_PreferCUVID;
    cuinfo.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;
    ctx->internal_error = CHECK_CU(cuvidCreateDecoder(&ctx->cudecoder, &cuinfo));
    if (ctx->internal_error < 0)
    if (!hwframe_ctx->pool) {
        hwframe_ctx->format = AV_PIX_FMT_CUDA;
        hwframe_ctx->sw_format = AV_PIX_FMT_NV12;
        hwframe_ctx->width = FFALIGN(avctx->coded_width, 32);
        hwframe_ctx->height = FFALIGN(avctx->coded_height, 32);
        if ((ctx->internal_error = av_hwframe_ctx_init(ctx->hwframe)) < 0) {
            av_log(avctx, AV_LOG_ERROR, "av_hwframe_ctx_init failed\n");
    return 1;