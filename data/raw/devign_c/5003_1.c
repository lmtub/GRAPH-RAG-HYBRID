static inline int parse_nal_units(AVCodecParserContext *s, const uint8_t *buf,
                           int buf_size, AVCodecContext *avctx)
{
    HEVCParserContext *ctx = s->priv_data;
    HEVCContext       *h   = &ctx->h;
    GetBitContext      *gb;
    SliceHeader        *sh = &h->sh;
    HEVCParamSets *ps = &h->ps;
    H2645Packet   *pkt = &ctx->pkt;
    const uint8_t *buf_end = buf + buf_size;
    int state = -1, i;
    H2645NAL *nal;
    int is_global = buf == avctx->extradata;
    if (!h->HEVClc)
        h->HEVClc = av_mallocz(sizeof(HEVCLocalContext));
    if (!h->HEVClc)
        return AVERROR(ENOMEM);
    gb = &h->HEVClc->gb;
    /* set some sane default values */
    s->pict_type         = AV_PICTURE_TYPE_I;
    s->key_frame         = 0;
    s->picture_structure = AV_PICTURE_STRUCTURE_UNKNOWN;
    h->avctx = avctx;
    if (!buf_size)
        return 0;
    if (pkt->nals_allocated < 1) {
        H2645NAL *tmp = av_realloc_array(pkt->nals, 1, sizeof(*tmp));
        if (!tmp)
            return AVERROR(ENOMEM);
        pkt->nals = tmp;
        memset(pkt->nals, 0, sizeof(*tmp));
        pkt->nals_allocated = 1;
    }
    nal = &pkt->nals[0];
    for (;;) {
        int src_length, consumed;
        int ret;
        buf = avpriv_find_start_code(buf, buf_end, &state);
        if (--buf + 2 >= buf_end)
            break;
        src_length = buf_end - buf;
        h->nal_unit_type = (*buf >> 1) & 0x3f;
        h->temporal_id   = (*(buf + 1) & 0x07) - 1;
        if (h->nal_unit_type <= NAL_CRA_NUT) {
            // Do not walk the whole buffer just to decode slice segment header
            if (src_length > 20)
                src_length = 20;
        }
        consumed = ff_h2645_extract_rbsp(buf, src_length, nal);
        if (consumed < 0)
            return consumed;
        ret = init_get_bits8(gb, nal->data + 2, nal->size);
        if (ret < 0)
            return ret;
        switch (h->nal_unit_type) {
        case NAL_VPS:
            ff_hevc_decode_nal_vps(gb, avctx, ps);
            break;
        case NAL_SPS:
            ff_hevc_decode_nal_sps(gb, avctx, ps, 1);
            break;
        case NAL_PPS:
            ff_hevc_decode_nal_pps(gb, avctx, ps);
            break;
        case NAL_SEI_PREFIX:
        case NAL_SEI_SUFFIX:
            ff_hevc_decode_nal_sei(h);
            break;
        case NAL_TRAIL_N:
        case NAL_TRAIL_R:
        case NAL_TSA_N:
        case NAL_TSA_R:
        case NAL_STSA_N:
        case NAL_STSA_R:
        case NAL_RADL_N:
        case NAL_RADL_R:
        case NAL_RASL_N:
        case NAL_RASL_R:
        case NAL_BLA_W_LP:
        case NAL_BLA_W_RADL:
        case NAL_BLA_N_LP:
        case NAL_IDR_W_RADL:
        case NAL_IDR_N_LP:
        case NAL_CRA_NUT:
            if (is_global) {
                av_log(avctx, AV_LOG_ERROR, "Invalid NAL unit: %d\n", h->nal_unit_type);
                return AVERROR_INVALIDDATA;
            }
            sh->first_slice_in_pic_flag = get_bits1(gb);
            s->picture_structure = h->picture_struct;
            s->field_order = h->picture_struct;
            if (IS_IRAP(h)) {
                s->key_frame = 1;
                sh->no_output_of_prior_pics_flag = get_bits1(gb);
            }
            sh->pps_id = get_ue_golomb(gb);
            if (sh->pps_id >= MAX_PPS_COUNT || !ps->pps_list[sh->pps_id]) {
                av_log(avctx, AV_LOG_ERROR, "PPS id out of range: %d\n", sh->pps_id);
                return AVERROR_INVALIDDATA;
            }
            ps->pps = (HEVCPPS*)ps->pps_list[sh->pps_id]->data;
            if (ps->pps->sps_id >= MAX_SPS_COUNT || !ps->sps_list[ps->pps->sps_id]) {
                av_log(avctx, AV_LOG_ERROR, "SPS id out of range: %d\n", ps->pps->sps_id);
                return AVERROR_INVALIDDATA;
            }
            if (ps->sps != (HEVCSPS*)ps->sps_list[ps->pps->sps_id]->data) {
                ps->sps = (HEVCSPS*)ps->sps_list[ps->pps->sps_id]->data;
                ps->vps = (HEVCVPS*)ps->vps_list[ps->sps->vps_id]->data;
            }
            if (!sh->first_slice_in_pic_flag) {
                int slice_address_length;
                if (ps->pps->dependent_slice_segments_enabled_flag)
                    sh->dependent_slice_segment_flag = get_bits1(gb);
                else
                    sh->dependent_slice_segment_flag = 0;
                slice_address_length = av_ceil_log2_c(ps->sps->ctb_width *
                                                      ps->sps->ctb_height);
                sh->slice_segment_addr = get_bitsz(gb, slice_address_length);
                if (sh->slice_segment_addr >= ps->sps->ctb_width * ps->sps->ctb_height) {
                    av_log(avctx, AV_LOG_ERROR, "Invalid slice segment address: %u.\n",
                           sh->slice_segment_addr);
                    return AVERROR_INVALIDDATA;
                }
            } else
                sh->dependent_slice_segment_flag = 0;
            if (sh->dependent_slice_segment_flag)
                break;
            for (i = 0; i < ps->pps->num_extra_slice_header_bits; i++)
                skip_bits(gb, 1); // slice_reserved_undetermined_flag[]
            sh->slice_type = get_ue_golomb(gb);
            if (!(sh->slice_type == I_SLICE || sh->slice_type == P_SLICE ||
                  sh->slice_type == B_SLICE)) {
                av_log(avctx, AV_LOG_ERROR, "Unknown slice type: %d.\n",
                       sh->slice_type);
                return AVERROR_INVALIDDATA;
            }
            s->pict_type = sh->slice_type == B_SLICE ? AV_PICTURE_TYPE_B :
                           sh->slice_type == P_SLICE ? AV_PICTURE_TYPE_P :
                                                       AV_PICTURE_TYPE_I;
            if (ps->pps->output_flag_present_flag)
                sh->pic_output_flag = get_bits1(gb);
            if (ps->sps->separate_colour_plane_flag)
                sh->colour_plane_id = get_bits(gb, 2);
            if (!IS_IDR(h)) {
                sh->pic_order_cnt_lsb = get_bits(gb, ps->sps->log2_max_poc_lsb);
                s->output_picture_number = h->poc = ff_hevc_compute_poc(h, sh->pic_order_cnt_lsb);
            } else
                s->output_picture_number = h->poc = 0;
            if (h->temporal_id == 0 &&
                h->nal_unit_type != NAL_TRAIL_N &&
                h->nal_unit_type != NAL_TSA_N &&
                h->nal_unit_type != NAL_STSA_N &&
                h->nal_unit_type != NAL_RADL_N &&
                h->nal_unit_type != NAL_RASL_N &&
                h->nal_unit_type != NAL_RADL_R &&
                h->nal_unit_type != NAL_RASL_R)
                h->pocTid0 = h->poc;
            return 0; /* no need to evaluate the rest */
        }
        buf += consumed;
    }
    /* didn't find a picture! */
    if (!is_global)
        av_log(h->avctx, AV_LOG_ERROR, "missing picture in access unit\n");
    return -1;
}