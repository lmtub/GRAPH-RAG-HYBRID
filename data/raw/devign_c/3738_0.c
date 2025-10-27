static int decode_nal_unit(HEVCContext *s, const HEVCNAL *nal)

{

    HEVCLocalContext *lc = &s->HEVClc;

    GetBitContext *gb    = &lc->gb;

    int ctb_addr_ts, ret;



    ret = init_get_bits8(gb, nal->data, nal->size);

    if (ret < 0)

        return ret;



    ret = hls_nal_unit(s);

    if (ret < 0) {

        av_log(s->avctx, AV_LOG_ERROR, "Invalid NAL unit %d, skipping.\n",

               s->nal_unit_type);

        goto fail;

    } else if (!ret)

        return 0;



    switch (s->nal_unit_type) {

    case NAL_VPS:

        ret = ff_hevc_decode_nal_vps(gb, s->avctx, &s->ps);

        if (ret < 0)

            goto fail;

        break;

    case NAL_SPS:

        ret = ff_hevc_decode_nal_sps(gb, s->avctx, &s->ps,

                                     s->apply_defdispwin);

        if (ret < 0)

            goto fail;

        break;

    case NAL_PPS:

        ret = ff_hevc_decode_nal_pps(gb, s->avctx, &s->ps);

        if (ret < 0)

            goto fail;

        break;

    case NAL_SEI_PREFIX:

    case NAL_SEI_SUFFIX:

        ret = ff_hevc_decode_nal_sei(s);

        if (ret < 0)

            goto fail;

        break;

    case NAL_TRAIL_R:

    case NAL_TRAIL_N:

    case NAL_TSA_N:

    case NAL_TSA_R:

    case NAL_STSA_N:

    case NAL_STSA_R:

    case NAL_BLA_W_LP:

    case NAL_BLA_W_RADL:

    case NAL_BLA_N_LP:

    case NAL_IDR_W_RADL:

    case NAL_IDR_N_LP:

    case NAL_CRA_NUT:

    case NAL_RADL_N:

    case NAL_RADL_R:

    case NAL_RASL_N:

    case NAL_RASL_R:

        ret = hls_slice_header(s);

        if (ret < 0)

            return ret;



        if (s->max_ra == INT_MAX) {

            if (s->nal_unit_type == NAL_CRA_NUT || IS_BLA(s)) {

                s->max_ra = s->poc;

            } else {

                if (IS_IDR(s))

                    s->max_ra = INT_MIN;

            }

        }



        if ((s->nal_unit_type == NAL_RASL_R || s->nal_unit_type == NAL_RASL_N) &&

            s->poc <= s->max_ra) {

            s->is_decoded = 0;

            break;

        } else {

            if (s->nal_unit_type == NAL_RASL_R && s->poc > s->max_ra)

                s->max_ra = INT_MIN;

        }



        if (s->sh.first_slice_in_pic_flag) {

            ret = hevc_frame_start(s);

            if (ret < 0)

                return ret;

        } else if (!s->ref) {

            av_log(s->avctx, AV_LOG_ERROR, "First slice in a frame missing.\n");

            goto fail;

        }



        if (s->nal_unit_type != s->first_nal_type) {

            av_log(s->avctx, AV_LOG_ERROR,

                   "Non-matching NAL types of the VCL NALUs: %d %d\n",

                   s->first_nal_type, s->nal_unit_type);

            return AVERROR_INVALIDDATA;

        }



        if (!s->sh.dependent_slice_segment_flag &&

            s->sh.slice_type != I_SLICE) {

            ret = ff_hevc_slice_rpl(s);

            if (ret < 0) {

                av_log(s->avctx, AV_LOG_WARNING,

                       "Error constructing the reference lists for the current slice.\n");

                goto fail;

            }

        }



        if (s->sh.first_slice_in_pic_flag && s->avctx->hwaccel) {

            ret = s->avctx->hwaccel->start_frame(s->avctx, NULL, 0);

            if (ret < 0)

                goto fail;

        }



        if (s->avctx->hwaccel) {

            ret = s->avctx->hwaccel->decode_slice(s->avctx, nal->raw_data, nal->raw_size);

            if (ret < 0)

                goto fail;

        } else {

            ctb_addr_ts = hls_slice_data(s);

            if (ctb_addr_ts >= (s->ps.sps->ctb_width * s->ps.sps->ctb_height)) {

                s->is_decoded = 1;

                if ((s->ps.pps->transquant_bypass_enable_flag ||

                     (s->ps.sps->pcm.loop_filter_disable_flag && s->ps.sps->pcm_enabled_flag)) &&

                    s->ps.sps->sao_enabled)

                    restore_tqb_pixels(s);

            }



            if (ctb_addr_ts < 0) {

                ret = ctb_addr_ts;

                goto fail;

            }

        }

        break;

    case NAL_EOS_NUT:

    case NAL_EOB_NUT:

        s->seq_decode = (s->seq_decode + 1) & 0xff;

        s->max_ra     = INT_MAX;

        break;

    case NAL_AUD:

    case NAL_FD_NUT:

        break;

    default:

        av_log(s->avctx, AV_LOG_INFO,

               "Skipping NAL unit %d\n", s->nal_unit_type);

    }



    return 0;

fail:

    if (s->avctx->err_recognition & AV_EF_EXPLODE)

        return ret;

    return 0;

}
