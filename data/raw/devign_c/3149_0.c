static int hevc_parse_slice_header(AVCodecParserContext *s, H2645NAL *nal,

                                   AVCodecContext *avctx)

{

    HEVCParserContext *ctx = s->priv_data;

    GetBitContext *gb = &nal->gb;



    HEVCPPS *pps;

    HEVCSPS *sps;

    unsigned int pps_id;



    get_bits1(gb);          // first slice in pic

    if (IS_IRAP_NAL(nal))

        get_bits1(gb);      // no output of prior pics



    pps_id = get_ue_golomb_long(gb);

    if (pps_id >= HEVC_MAX_PPS_COUNT || !ctx->ps.pps_list[pps_id]) {

        av_log(avctx, AV_LOG_ERROR, "PPS id out of range: %d\n", pps_id);

        return AVERROR_INVALIDDATA;

    }

    pps = (HEVCPPS*)ctx->ps.pps_list[pps_id]->data;

    sps = (HEVCSPS*)ctx->ps.sps_list[pps->sps_id]->data;



    /* export the stream parameters */

    s->coded_width  = sps->width;

    s->coded_height = sps->height;

    s->width        = sps->output_width;

    s->height       = sps->output_height;

    s->format       = sps->pix_fmt;

    avctx->profile  = sps->ptl.general_ptl.profile_idc;

    avctx->level    = sps->ptl.general_ptl.level_idc;



    /* ignore the rest for now*/



    return 0;

}
