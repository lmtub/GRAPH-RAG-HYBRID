static av_cold int X264_init(AVCodecContext *avctx)

{

    X264Context *x4 = avctx->priv_data;

    AVCPBProperties *cpb_props;



#if CONFIG_LIBX262_ENCODER

    if (avctx->codec_id == AV_CODEC_ID_MPEG2VIDEO) {

        x4->params.b_mpeg2 = 1;

        x264_param_default_mpeg2(&x4->params);

    } else

#else

    x264_param_default(&x4->params);

#endif



    x4->params.b_deblocking_filter         = avctx->flags & AV_CODEC_FLAG_LOOP_FILTER;



    if (x4->preset || x4->tune)

        if (x264_param_default_preset(&x4->params, x4->preset, x4->tune) < 0) {

            av_log(avctx, AV_LOG_ERROR, "Error setting preset/tune %s/%s.\n", x4->preset, x4->tune);

            return AVERROR(EINVAL);

        }



    if (avctx->level > 0)

        x4->params.i_level_idc = avctx->level;



    x4->params.pf_log               = X264_log;

    x4->params.p_log_private        = avctx;

    x4->params.i_log_level          = X264_LOG_DEBUG;

    x4->params.i_csp                = convert_pix_fmt(avctx->pix_fmt);



    if (avctx->bit_rate) {

        x4->params.rc.i_bitrate   = avctx->bit_rate / 1000;

        x4->params.rc.i_rc_method = X264_RC_ABR;

    }

    x4->params.rc.i_vbv_buffer_size = avctx->rc_buffer_size / 1000;

    x4->params.rc.i_vbv_max_bitrate = avctx->rc_max_rate    / 1000;

    x4->params.rc.b_stat_write      = avctx->flags & AV_CODEC_FLAG_PASS1;

    if (avctx->flags & AV_CODEC_FLAG_PASS2) {

        x4->params.rc.b_stat_read = 1;

    } else {

        if (x4->crf >= 0) {

            x4->params.rc.i_rc_method   = X264_RC_CRF;

            x4->params.rc.f_rf_constant = x4->crf;

        } else if (x4->cqp >= 0) {

            x4->params.rc.i_rc_method   = X264_RC_CQP;

            x4->params.rc.i_qp_constant = x4->cqp;

        }



        if (x4->crf_max >= 0)

            x4->params.rc.f_rf_constant_max = x4->crf_max;

    }



    if (avctx->rc_buffer_size && avctx->rc_initial_buffer_occupancy > 0 &&

        (avctx->rc_initial_buffer_occupancy <= avctx->rc_buffer_size)) {

        x4->params.rc.f_vbv_buffer_init =

            (float)avctx->rc_initial_buffer_occupancy / avctx->rc_buffer_size;

    }



    if (avctx->i_quant_factor > 0)

        x4->params.rc.f_ip_factor         = 1 / fabs(avctx->i_quant_factor);

    x4->params.rc.f_pb_factor             = avctx->b_quant_factor;

    x4->params.analyse.i_chroma_qp_offset = avctx->chromaoffset;



    if (avctx->gop_size >= 0)

        x4->params.i_keyint_max         = avctx->gop_size;

    if (avctx->max_b_frames >= 0)

        x4->params.i_bframe             = avctx->max_b_frames;

    if (avctx->scenechange_threshold >= 0)

        x4->params.i_scenecut_threshold = avctx->scenechange_threshold;

    if (avctx->qmin >= 0)

        x4->params.rc.i_qp_min          = avctx->qmin;

    if (avctx->qmax >= 0)

        x4->params.rc.i_qp_max          = avctx->qmax;

    if (avctx->max_qdiff >= 0)

        x4->params.rc.i_qp_step         = avctx->max_qdiff;

    if (avctx->qblur >= 0)

        x4->params.rc.f_qblur           = avctx->qblur;     /* temporally blur quants */

    if (avctx->qcompress >= 0)

        x4->params.rc.f_qcompress       = avctx->qcompress; /* 0.0 => cbr, 1.0 => constant qp */

    if (avctx->refs >= 0)

        x4->params.i_frame_reference    = avctx->refs;

    if (avctx->trellis >= 0)

        x4->params.analyse.i_trellis    = avctx->trellis;

    if (avctx->me_range >= 0)

        x4->params.analyse.i_me_range   = avctx->me_range;

    if (avctx->noise_reduction >= 0)

        x4->params.analyse.i_noise_reduction = avctx->noise_reduction;

    if (avctx->me_subpel_quality >= 0)

        x4->params.analyse.i_subpel_refine   = avctx->me_subpel_quality;

    if (avctx->b_frame_strategy >= 0)

        x4->params.i_bframe_adaptive = avctx->b_frame_strategy;

    if (avctx->keyint_min >= 0)

        x4->params.i_keyint_min = avctx->keyint_min;

#if FF_API_CODER_TYPE

FF_DISABLE_DEPRECATION_WARNINGS

    if (avctx->coder_type >= 0)

        x4->coder = avctx->coder_type == FF_CODER_TYPE_AC;

FF_ENABLE_DEPRECATION_WARNINGS

#endif

    if (avctx->me_cmp >= 0)

        x4->params.analyse.b_chroma_me = avctx->me_cmp & FF_CMP_CHROMA;



    if (x4->aq_mode >= 0)

        x4->params.rc.i_aq_mode = x4->aq_mode;

    if (x4->aq_strength >= 0)

        x4->params.rc.f_aq_strength = x4->aq_strength;

    PARSE_X264_OPT("psy-rd", psy_rd);

    PARSE_X264_OPT("deblock", deblock);

    PARSE_X264_OPT("partitions", partitions);

    PARSE_X264_OPT("stats", stats);

    if (x4->psy >= 0)

        x4->params.analyse.b_psy  = x4->psy;

    if (x4->rc_lookahead >= 0)

        x4->params.rc.i_lookahead = x4->rc_lookahead;

    if (x4->weightp >= 0)

        x4->params.analyse.i_weighted_pred = x4->weightp;

    if (x4->weightb >= 0)

        x4->params.analyse.b_weighted_bipred = x4->weightb;

    if (x4->cplxblur >= 0)

        x4->params.rc.f_complexity_blur = x4->cplxblur;



    if (x4->ssim >= 0)

        x4->params.analyse.b_ssim = x4->ssim;

    if (x4->intra_refresh >= 0)

        x4->params.b_intra_refresh = x4->intra_refresh;

    if (x4->bluray_compat >= 0) {

        x4->params.b_bluray_compat = x4->bluray_compat;

        x4->params.b_vfr_input = 0;

    }

    if (x4->b_bias != INT_MIN)

        x4->params.i_bframe_bias              = x4->b_bias;

    if (x4->b_pyramid >= 0)

        x4->params.i_bframe_pyramid = x4->b_pyramid;

    if (x4->mixed_refs >= 0)

        x4->params.analyse.b_mixed_references = x4->mixed_refs;

    if (x4->dct8x8 >= 0)

        x4->params.analyse.b_transform_8x8    = x4->dct8x8;

    if (x4->fast_pskip >= 0)

        x4->params.analyse.b_fast_pskip       = x4->fast_pskip;

    if (x4->aud >= 0)

        x4->params.b_aud                      = x4->aud;

    if (x4->mbtree >= 0)

        x4->params.rc.b_mb_tree               = x4->mbtree;

    if (x4->direct_pred >= 0)

        x4->params.analyse.i_direct_mv_pred   = x4->direct_pred;



    if (x4->slice_max_size >= 0)

        x4->params.i_slice_max_size =  x4->slice_max_size;



    if (x4->fastfirstpass)

        x264_param_apply_fastfirstpass(&x4->params);



    if (x4->nal_hrd >= 0)

        x4->params.i_nal_hrd = x4->nal_hrd;



    if (x4->motion_est >= 0) {

        x4->params.analyse.i_me_method = x4->motion_est;

#if FF_API_MOTION_EST

FF_DISABLE_DEPRECATION_WARNINGS

    } else {

        if (avctx->me_method == ME_EPZS)

            x4->params.analyse.i_me_method = X264_ME_DIA;

        else if (avctx->me_method == ME_HEX)

            x4->params.analyse.i_me_method = X264_ME_HEX;

        else if (avctx->me_method == ME_UMH)

            x4->params.analyse.i_me_method = X264_ME_UMH;

        else if (avctx->me_method == ME_FULL)

            x4->params.analyse.i_me_method = X264_ME_ESA;

        else if (avctx->me_method == ME_TESA)

            x4->params.analyse.i_me_method = X264_ME_TESA;

FF_ENABLE_DEPRECATION_WARNINGS

#endif

    }



    if (x4->coder >= 0)

        x4->params.b_cabac = x4->coder;



    if (x4->profile)

        if (x264_param_apply_profile(&x4->params, x4->profile) < 0) {

            av_log(avctx, AV_LOG_ERROR, "Error setting profile %s.\n", x4->profile);

            return AVERROR(EINVAL);

        }



    x4->params.i_width          = avctx->width;

    x4->params.i_height         = avctx->height;

    x4->params.vui.i_sar_width  = avctx->sample_aspect_ratio.num;

    x4->params.vui.i_sar_height = avctx->sample_aspect_ratio.den;

    x4->params.i_fps_num = x4->params.i_timebase_den = avctx->time_base.den;

    x4->params.i_fps_den = x4->params.i_timebase_num = avctx->time_base.num;



    x4->params.analyse.b_psnr = avctx->flags & AV_CODEC_FLAG_PSNR;



    x4->params.i_threads      = avctx->thread_count;

    if (avctx->thread_type)

        x4->params.b_sliced_threads = avctx->thread_type == FF_THREAD_SLICE;



    x4->params.b_interlaced   = avctx->flags & AV_CODEC_FLAG_INTERLACED_DCT;



    x4->params.b_open_gop     = !(avctx->flags & AV_CODEC_FLAG_CLOSED_GOP);



    x4->params.i_slice_count  = avctx->slices;



    x4->params.vui.b_fullrange = avctx->pix_fmt == AV_PIX_FMT_YUVJ420P ||

                                 avctx->pix_fmt == AV_PIX_FMT_YUVJ422P ||

                                 avctx->pix_fmt == AV_PIX_FMT_YUVJ444P ||

                                 avctx->color_range == AVCOL_RANGE_JPEG;



    // x264 validates the values internally

    x4->params.vui.i_colorprim = avctx->color_primaries;

    x4->params.vui.i_transfer  = avctx->color_trc;

    x4->params.vui.i_colmatrix = avctx->colorspace;



    if (avctx->flags & AV_CODEC_FLAG_GLOBAL_HEADER)

        x4->params.b_repeat_headers = 0;



    if (x4->x264_params) {

        AVDictionary *dict    = NULL;

        AVDictionaryEntry *en = NULL;



        if (!av_dict_parse_string(&dict, x4->x264_params, "=", ":", 0)) {

            while ((en = av_dict_get(dict, "", en, AV_DICT_IGNORE_SUFFIX))) {

                if (x264_param_parse(&x4->params, en->key, en->value) < 0)

                    av_log(avctx, AV_LOG_WARNING,

                           "Error parsing option '%s = %s'.\n",

                            en->key, en->value);

            }



            av_dict_free(&dict);

        }

    }



    // update AVCodecContext with x264 parameters

    avctx->has_b_frames = x4->params.i_bframe ?

        x4->params.i_bframe_pyramid ? 2 : 1 : 0;

    if (avctx->max_b_frames < 0)

        avctx->max_b_frames = 0;



    avctx->bit_rate = x4->params.rc.i_bitrate*1000;



    x4->enc = x264_encoder_open(&x4->params);

    if (!x4->enc)

        return AVERROR_UNKNOWN;



    if (avctx->flags & AV_CODEC_FLAG_GLOBAL_HEADER) {

        x264_nal_t *nal;

        uint8_t *p;

        int nnal, s, i;



        s = x264_encoder_headers(x4->enc, &nal, &nnal);

        avctx->extradata = p = av_mallocz(s + AV_INPUT_BUFFER_PADDING_SIZE);

        if (!p)

            return AVERROR(ENOMEM);



        for (i = 0; i < nnal; i++) {

            /* Don't put the SEI in extradata. */

            if (nal[i].i_type == NAL_SEI) {

                av_log(avctx, AV_LOG_INFO, "%s\n", nal[i].p_payload+25);

                x4->sei_size = nal[i].i_payload;

                x4->sei      = av_malloc(x4->sei_size);

                if (!x4->sei)

                    return AVERROR(ENOMEM);

                memcpy(x4->sei, nal[i].p_payload, nal[i].i_payload);

                continue;

            }

            memcpy(p, nal[i].p_payload, nal[i].i_payload);

            p += nal[i].i_payload;

        }

        avctx->extradata_size = p - avctx->extradata;

    }



    cpb_props = ff_add_cpb_side_data(avctx);

    if (!cpb_props)

        return AVERROR(ENOMEM);

    cpb_props->buffer_size = x4->params.rc.i_vbv_buffer_size * 1000;

    cpb_props->max_bitrate = x4->params.rc.i_vbv_max_bitrate * 1000;

    cpb_props->avg_bitrate = x4->params.rc.i_bitrate         * 1000;



    return 0;

}
