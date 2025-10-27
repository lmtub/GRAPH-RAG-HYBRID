static int nvdec_vc1_end_frame(AVCodecContext *avctx)

{

    NVDECContext *ctx = avctx->internal->hwaccel_priv_data;

    int ret = ff_nvdec_end_frame(avctx);

    ctx->bitstream = NULL;

    return ret;

}
