int ff_vdpau_common_init(AVCodecContext *avctx, VdpDecoderProfile profile,
                         int level)
{
    VDPAUHWContext *hwctx = avctx->hwaccel_context;
    VDPAUContext *vdctx = avctx->internal->hwaccel_priv_data;
    VdpVideoSurfaceQueryCapabilities *surface_query_caps;
    VdpDecoderQueryCapabilities *decoder_query_caps;
    VdpDecoderCreate *create;
    VdpGetInformationString *info;
    const char *info_string;
    void *func;
    VdpStatus status;
    VdpBool supported;
    uint32_t max_level, max_mb, max_width, max_height;
    VdpChromaType type;
    uint32_t width;
    uint32_t height;
    vdctx->width            = UINT32_MAX;
    vdctx->height           = UINT32_MAX;
    if (av_vdpau_get_surface_parameters(avctx, &type, &width, &height))
        return AVERROR(ENOSYS);
    if (hwctx) {
        hwctx->reset            = 0;
        if (hwctx->context.decoder != VDP_INVALID_HANDLE) {
            vdctx->decoder = hwctx->context.decoder;
            vdctx->render  = hwctx->context.render;
            vdctx->device  = VDP_INVALID_HANDLE;
            return 0; /* Decoder created by user */
        vdctx->device           = hwctx->device;
        vdctx->get_proc_address = hwctx->get_proc_address;
        if (hwctx->flags & AV_HWACCEL_FLAG_IGNORE_LEVEL)
            level = 0;
        if (!(hwctx->flags & AV_HWACCEL_FLAG_ALLOW_HIGH_DEPTH) &&
            type != VDP_CHROMA_TYPE_420)
            return AVERROR(ENOSYS);
    } else {
        AVHWFramesContext *frames_ctx = NULL;
        AVVDPAUDeviceContext *dev_ctx;
        // We assume the hw_frames_ctx always survives until ff_vdpau_common_uninit
        // is called. This holds true as the user is not allowed to touch
        // hw_device_ctx, or hw_frames_ctx after get_format (and ff_get_format
        // itself also uninits before unreffing hw_frames_ctx).
        if (avctx->hw_frames_ctx) {
            frames_ctx = (AVHWFramesContext*)avctx->hw_frames_ctx->data;
        } else if (avctx->hw_device_ctx) {
            int ret;
            avctx->hw_frames_ctx = av_hwframe_ctx_alloc(avctx->hw_device_ctx);
            if (!avctx->hw_frames_ctx)
                return AVERROR(ENOMEM);
            frames_ctx            = (AVHWFramesContext*)avctx->hw_frames_ctx->data;
            frames_ctx->format    = AV_PIX_FMT_VDPAU;
            frames_ctx->sw_format = avctx->sw_pix_fmt;
            frames_ctx->width     = avctx->coded_width;
            frames_ctx->height    = avctx->coded_height;
            ret = av_hwframe_ctx_init(avctx->hw_frames_ctx);
            if (ret < 0) {
                av_buffer_unref(&avctx->hw_frames_ctx);
                return ret;
        if (!frames_ctx) {
            av_log(avctx, AV_LOG_ERROR, "A hardware frames context is "
                   "required for VDPAU decoding.\n");
            return AVERROR(EINVAL);
        dev_ctx = frames_ctx->device_ctx->hwctx;
        vdctx->device           = dev_ctx->device;
        vdctx->get_proc_address = dev_ctx->get_proc_address;
        if (avctx->hwaccel_flags & AV_HWACCEL_FLAG_IGNORE_LEVEL)
            level = 0;
    if (level < 0)
                                     VDP_FUNC_ID_VIDEO_SURFACE_QUERY_CAPABILITIES,
        surface_query_caps = func;
    status = surface_query_caps(vdctx->device, type, &supported,
                                &max_width, &max_height);
    if (supported != VDP_TRUE ||
        max_width < width || max_height < height)
                                     VDP_FUNC_ID_DECODER_QUERY_CAPABILITIES,
        decoder_query_caps = func;
    status = decoder_query_caps(vdctx->device, profile, &supported, &max_level,
                                &max_mb, &max_width, &max_height);
#ifdef VDP_DECODER_PROFILE_H264_CONSTRAINED_BASELINE
    if ((status != VDP_STATUS_OK || supported != VDP_TRUE) && profile == VDP_DECODER_PROFILE_H264_CONSTRAINED_BASELINE) {
        profile = VDP_DECODER_PROFILE_H264_MAIN;
        status = decoder_query_caps(vdctx->device, profile, &supported,
                                    &max_level, &max_mb,
                                    &max_width, &max_height);
#endif
    if (supported != VDP_TRUE || max_level < level ||
        max_width < width || max_height < height)
    status = vdctx->get_proc_address(vdctx->device, VDP_FUNC_ID_DECODER_CREATE,
        create = func;
    status = vdctx->get_proc_address(vdctx->device, VDP_FUNC_ID_DECODER_RENDER,
        vdctx->render = func;
    status = create(vdctx->device, profile, width, height, avctx->refs,
                    &vdctx->decoder);
    if (status == VDP_STATUS_OK) {
        vdctx->width  = avctx->coded_width;
        vdctx->height = avctx->coded_height;