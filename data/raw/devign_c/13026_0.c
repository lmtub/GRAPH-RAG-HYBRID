static int nvdec_hevc_decode_init(AVCodecContext *avctx)

{

    const HEVCContext *s = avctx->priv_data;

    const HEVCSPS *sps = s->ps.sps;

    return ff_nvdec_decode_init(avctx, sps->temporal_layer[sps->max_sub_layers - 1].max_dec_pic_buffering + 1);

}
