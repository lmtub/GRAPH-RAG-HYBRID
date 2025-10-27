MAKE_ACCESSORS(AVVDPAUContext, vdpau_hwaccel, AVVDPAU_Render2, render2)
int ff_vdpau_common_init(AVCodecContext *avctx, VdpDecoderProfile profile,
                         int level)
{
    VDPAUHWContext *hwctx = avctx->hwaccel_context;
    VDPAUContext *vdctx = avctx->internal->hwaccel_priv_data;
    VdpDecoderCreate *create;
    void *func;
    VdpStatus status;
    /* See vdpau/vdpau.h for alignment constraints. */
    uint32_t width  = (avctx->coded_width + 1) & ~1;
    uint32_t height = (avctx->coded_height + 3) & ~3;
    if (hwctx->context.decoder != VDP_INVALID_HANDLE) {
        vdctx->decoder = hwctx->context.decoder;
        vdctx->render  = hwctx->context.render;
        return 0; /* Decoder created by user */
    vdctx->device           = hwctx->device;
    vdctx->get_proc_address = hwctx->get_proc_address;
    status = vdctx->get_proc_address(vdctx->device, VDP_FUNC_ID_DECODER_CREATE,
                                     &func);
    if (status != VDP_STATUS_OK)
        return vdpau_error(status);
    else
        create = func;
    status = vdctx->get_proc_address(vdctx->device, VDP_FUNC_ID_DECODER_RENDER,
                                     &func);
    if (status != VDP_STATUS_OK)
        return vdpau_error(status);
    else
        vdctx->render = func;
    status = create(vdctx->device, profile, width, height, avctx->refs,
                    &vdctx->decoder);
    return vdpau_error(status);