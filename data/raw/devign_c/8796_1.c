av_cold int ff_mpv_encode_init(AVCodecContext *avctx)

{

    MpegEncContext *s = avctx->priv_data;

    int i, ret, format_supported;



    mpv_encode_defaults(s);



    switch (avctx->codec_id) {

    case AV_CODEC_ID_MPEG2VIDEO:

        if (avctx->pix_fmt != AV_PIX_FMT_YUV420P &&

            avctx->pix_fmt != AV_PIX_FMT_YUV422P) {

            av_log(avctx, AV_LOG_ERROR,

                   "only YUV420 and YUV422 are supported\n");

            return -1;

        }

        break;

    case AV_CODEC_ID_MJPEG:

        format_supported = 0;

        /* JPEG color space */

        if (avctx->pix_fmt == AV_PIX_FMT_YUVJ420P ||

            avctx->pix_fmt == AV_PIX_FMT_YUVJ422P ||

            (avctx->color_range == AVCOL_RANGE_JPEG &&

             (avctx->pix_fmt == AV_PIX_FMT_YUV420P ||

              avctx->pix_fmt == AV_PIX_FMT_YUV422P)))

            format_supported = 1;

        /* MPEG color space */

        else if (avctx->strict_std_compliance <= FF_COMPLIANCE_UNOFFICIAL &&

                 (avctx->pix_fmt == AV_PIX_FMT_YUV420P ||

                  avctx->pix_fmt == AV_PIX_FMT_YUV422P))

            format_supported = 1;



        if (!format_supported) {

            av_log(avctx, AV_LOG_ERROR, "colorspace not supported in jpeg\n");

            return -1;

        }

        break;

    default:

        if (avctx->pix_fmt != AV_PIX_FMT_YUV420P) {

            av_log(avctx, AV_LOG_ERROR, "only YUV420 is supported\n");

            return -1;

        }

    }



    switch (avctx->pix_fmt) {

    case AV_PIX_FMT_YUVJ422P:

    case AV_PIX_FMT_YUV422P:

        s->chroma_format = CHROMA_422;

        break;

    case AV_PIX_FMT_YUVJ420P:

    case AV_PIX_FMT_YUV420P:

    default:

        s->chroma_format = CHROMA_420;

        break;

    }



    s->bit_rate = avctx->bit_rate;

    s->width    = avctx->width;

    s->height   = avctx->height;

    if (avctx->gop_size > 600 &&

        avctx->strict_std_compliance > FF_COMPLIANCE_EXPERIMENTAL) {

        av_log(avctx, AV_LOG_ERROR,

               "Warning keyframe interval too large! reducing it ...\n");

        avctx->gop_size = 600;

    }

    s->gop_size     = avctx->gop_size;

    s->avctx        = avctx;

    if (avctx->max_b_frames > MAX_B_FRAMES) {

        av_log(avctx, AV_LOG_ERROR, "Too many B-frames requested, maximum "

               "is %d.\n", MAX_B_FRAMES);

    }

    s->max_b_frames = avctx->max_b_frames;

    s->codec_id     = avctx->codec->id;

    s->strict_std_compliance = avctx->strict_std_compliance;

    s->quarter_sample     = (avctx->flags & AV_CODEC_FLAG_QPEL) != 0;

    s->mpeg_quant         = avctx->mpeg_quant;

    s->rtp_mode           = !!avctx->rtp_payload_size;

    s->intra_dc_precision = avctx->intra_dc_precision;

    s->user_specified_pts = AV_NOPTS_VALUE;



    if (s->gop_size <= 1) {

        s->intra_only = 1;

        s->gop_size   = 12;

    } else {

        s->intra_only = 0;

    }



#if FF_API_MOTION_EST

FF_DISABLE_DEPRECATION_WARNINGS

    s->me_method = avctx->me_method;

FF_ENABLE_DEPRECATION_WARNINGS

#endif



    /* Fixed QSCALE */

    s->fixed_qscale = !!(avctx->flags & AV_CODEC_FLAG_QSCALE);



#if FF_API_MPV_OPT

    FF_DISABLE_DEPRECATION_WARNINGS

    if (avctx->border_masking != 0.0)

        s->border_masking = avctx->border_masking;

    FF_ENABLE_DEPRECATION_WARNINGS

#endif



    s->adaptive_quant = (s->avctx->lumi_masking ||

                         s->avctx->dark_masking ||

                         s->avctx->temporal_cplx_masking ||

                         s->avctx->spatial_cplx_masking  ||

                         s->avctx->p_masking      ||

                         s->border_masking ||

                         (s->mpv_flags & FF_MPV_FLAG_QP_RD)) &&

                        !s->fixed_qscale;



    s->loop_filter = !!(s->avctx->flags & AV_CODEC_FLAG_LOOP_FILTER);



    if (avctx->rc_max_rate && !avctx->rc_buffer_size) {

        av_log(avctx, AV_LOG_ERROR,

               "a vbv buffer size is needed, "

               "for encoding with a maximum bitrate\n");

        return -1;

    }



    if (avctx->rc_min_rate && avctx->rc_max_rate != avctx->rc_min_rate) {

        av_log(avctx, AV_LOG_INFO,

               "Warning min_rate > 0 but min_rate != max_rate isn't recommended!\n");

    }



    if (avctx->rc_min_rate && avctx->rc_min_rate > avctx->bit_rate) {

        av_log(avctx, AV_LOG_ERROR, "bitrate below min bitrate\n");

        return -1;

    }



    if (avctx->rc_max_rate && avctx->rc_max_rate < avctx->bit_rate) {

        av_log(avctx, AV_LOG_INFO, "bitrate above max bitrate\n");

        return -1;

    }



    if (avctx->rc_max_rate &&

        avctx->rc_max_rate == avctx->bit_rate &&

        avctx->rc_max_rate != avctx->rc_min_rate) {

        av_log(avctx, AV_LOG_INFO,

               "impossible bitrate constraints, this will fail\n");

    }



    if (avctx->rc_buffer_size &&

        avctx->bit_rate * (int64_t)avctx->time_base.num >

            avctx->rc_buffer_size * (int64_t)avctx->time_base.den) {

        av_log(avctx, AV_LOG_ERROR, "VBV buffer too small for bitrate\n");

        return -1;

    }



    if (!s->fixed_qscale &&

        avctx->bit_rate * av_q2d(avctx->time_base) >

            avctx->bit_rate_tolerance) {

        av_log(avctx, AV_LOG_ERROR,

               "bitrate tolerance too small for bitrate\n");

        return -1;

    }



    if (s->avctx->rc_max_rate &&

        s->avctx->rc_min_rate == s->avctx->rc_max_rate &&

        (s->codec_id == AV_CODEC_ID_MPEG1VIDEO ||

         s->codec_id == AV_CODEC_ID_MPEG2VIDEO) &&

        90000LL * (avctx->rc_buffer_size - 1) >

            s->avctx->rc_max_rate * 0xFFFFLL) {

        av_log(avctx, AV_LOG_INFO,

               "Warning vbv_delay will be set to 0xFFFF (=VBR) as the "

               "specified vbv buffer is too large for the given bitrate!\n");

    }



    if ((s->avctx->flags & AV_CODEC_FLAG_4MV) && s->codec_id != AV_CODEC_ID_MPEG4 &&

        s->codec_id != AV_CODEC_ID_H263 && s->codec_id != AV_CODEC_ID_H263P &&

        s->codec_id != AV_CODEC_ID_FLV1) {

        av_log(avctx, AV_LOG_ERROR, "4MV not supported by codec\n");

        return -1;

    }



    if (s->obmc && s->avctx->mb_decision != FF_MB_DECISION_SIMPLE) {

        av_log(avctx, AV_LOG_ERROR,

               "OBMC is only supported with simple mb decision\n");

        return -1;

    }



    if (s->quarter_sample && s->codec_id != AV_CODEC_ID_MPEG4) {

        av_log(avctx, AV_LOG_ERROR, "qpel not supported by codec\n");

        return -1;

    }



    if (s->max_b_frames                    &&

        s->codec_id != AV_CODEC_ID_MPEG4      &&

        s->codec_id != AV_CODEC_ID_MPEG1VIDEO &&

        s->codec_id != AV_CODEC_ID_MPEG2VIDEO) {

        av_log(avctx, AV_LOG_ERROR, "b frames not supported by codec\n");

        return -1;

    }



    if ((s->codec_id == AV_CODEC_ID_MPEG4 ||

         s->codec_id == AV_CODEC_ID_H263  ||

         s->codec_id == AV_CODEC_ID_H263P) &&

        (avctx->sample_aspect_ratio.num > 255 ||

         avctx->sample_aspect_ratio.den > 255)) {

        av_log(avctx, AV_LOG_ERROR,

               "Invalid pixel aspect ratio %i/%i, limit is 255/255\n",

               avctx->sample_aspect_ratio.num, avctx->sample_aspect_ratio.den);

        return -1;

    }



    if ((s->avctx->flags & (AV_CODEC_FLAG_INTERLACED_DCT | AV_CODEC_FLAG_INTERLACED_ME)) &&

        s->codec_id != AV_CODEC_ID_MPEG4 && s->codec_id != AV_CODEC_ID_MPEG2VIDEO) {

        av_log(avctx, AV_LOG_ERROR, "interlacing not supported by codec\n");

        return -1;

    }



    // FIXME mpeg2 uses that too

    if (s->mpeg_quant && s->codec_id != AV_CODEC_ID_MPEG4) {

        av_log(avctx, AV_LOG_ERROR,

               "mpeg2 style quantization not supported by codec\n");

        return -1;

    }



    if ((s->mpv_flags & FF_MPV_FLAG_CBP_RD) && !avctx->trellis) {

        av_log(avctx, AV_LOG_ERROR, "CBP RD needs trellis quant\n");

        return -1;

    }



    if ((s->mpv_flags & FF_MPV_FLAG_QP_RD) &&

        s->avctx->mb_decision != FF_MB_DECISION_RD) {

        av_log(avctx, AV_LOG_ERROR, "QP RD needs mbd=2\n");

        return -1;

    }



    if (s->avctx->scenechange_threshold < 1000000000 &&

        (s->avctx->flags & AV_CODEC_FLAG_CLOSED_GOP)) {

        av_log(avctx, AV_LOG_ERROR,

               "closed gop with scene change detection are not supported yet, "

               "set threshold to 1000000000\n");

        return -1;

    }



    if (s->avctx->flags & AV_CODEC_FLAG_LOW_DELAY) {

        if (s->codec_id != AV_CODEC_ID_MPEG2VIDEO) {

            av_log(avctx, AV_LOG_ERROR,

                  "low delay forcing is only available for mpeg2\n");

            return -1;

        }

        if (s->max_b_frames != 0) {

            av_log(avctx, AV_LOG_ERROR,

                   "b frames cannot be used with low delay\n");

            return -1;

        }

    }



    if (s->q_scale_type == 1) {

        if (avctx->qmax > 12) {

            av_log(avctx, AV_LOG_ERROR,

                   "non linear quant only supports qmax <= 12 currently\n");

            return -1;

        }

    }



    if (avctx->slices > 1 &&

        (avctx->codec_id == AV_CODEC_ID_FLV1 || avctx->codec_id == AV_CODEC_ID_H261)) {

        av_log(avctx, AV_LOG_ERROR, "Multiple slices are not supported by this codec\n");

        return AVERROR(EINVAL);

    }



    if (s->avctx->thread_count > 1         &&

        s->codec_id != AV_CODEC_ID_MPEG4      &&

        s->codec_id != AV_CODEC_ID_MPEG1VIDEO &&

        s->codec_id != AV_CODEC_ID_MPEG2VIDEO &&

        (s->codec_id != AV_CODEC_ID_H263P)) {

        av_log(avctx, AV_LOG_ERROR,

               "multi threaded encoding not supported by codec\n");

        return -1;

    }



    if (s->avctx->thread_count < 1) {

        av_log(avctx, AV_LOG_ERROR,

               "automatic thread number detection not supported by codec,"

               "patch welcome\n");

        return -1;

    }



    if (s->avctx->thread_count > 1)

        s->rtp_mode = 1;



    if (!avctx->time_base.den || !avctx->time_base.num) {

        av_log(avctx, AV_LOG_ERROR, "framerate not set\n");

        return -1;

    }



    if (avctx->b_frame_strategy && (avctx->flags & AV_CODEC_FLAG_PASS2)) {

        av_log(avctx, AV_LOG_INFO,

               "notice: b_frame_strategy only affects the first pass\n");

        avctx->b_frame_strategy = 0;

    }



    i = av_gcd(avctx->time_base.den, avctx->time_base.num);

    if (i > 1) {

        av_log(avctx, AV_LOG_INFO, "removing common factors from framerate\n");

        avctx->time_base.den /= i;

        avctx->time_base.num /= i;

        //return -1;

    }



    if (s->mpeg_quant || s->codec_id == AV_CODEC_ID_MPEG1VIDEO ||

        s->codec_id == AV_CODEC_ID_MPEG2VIDEO || s->codec_id == AV_CODEC_ID_MJPEG) {

        // (a + x * 3 / 8) / x

        s->intra_quant_bias = 3 << (QUANT_BIAS_SHIFT - 3);

        s->inter_quant_bias = 0;

    } else {

        s->intra_quant_bias = 0;

        // (a - x / 4) / x

        s->inter_quant_bias = -(1 << (QUANT_BIAS_SHIFT - 2));

    }



#if FF_API_QUANT_BIAS

FF_DISABLE_DEPRECATION_WARNINGS

    if (avctx->intra_quant_bias != FF_DEFAULT_QUANT_BIAS)

        s->intra_quant_bias = avctx->intra_quant_bias;

    if (avctx->inter_quant_bias != FF_DEFAULT_QUANT_BIAS)

        s->inter_quant_bias = avctx->inter_quant_bias;

FF_ENABLE_DEPRECATION_WARNINGS

#endif



    if (avctx->codec_id == AV_CODEC_ID_MPEG4 &&

        s->avctx->time_base.den > (1 << 16) - 1) {

        av_log(avctx, AV_LOG_ERROR,

               "timebase %d/%d not supported by MPEG 4 standard, "

               "the maximum admitted value for the timebase denominator "

               "is %d\n", s->avctx->time_base.num, s->avctx->time_base.den,

               (1 << 16) - 1);

        return -1;

    }

    s->time_increment_bits = av_log2(s->avctx->time_base.den - 1) + 1;



    switch (avctx->codec->id) {

    case AV_CODEC_ID_MPEG1VIDEO:

        s->out_format = FMT_MPEG1;

        s->low_delay  = !!(s->avctx->flags & AV_CODEC_FLAG_LOW_DELAY);

        avctx->delay  = s->low_delay ? 0 : (s->max_b_frames + 1);

        break;

    case AV_CODEC_ID_MPEG2VIDEO:

        s->out_format = FMT_MPEG1;

        s->low_delay  = !!(s->avctx->flags & AV_CODEC_FLAG_LOW_DELAY);

        avctx->delay  = s->low_delay ? 0 : (s->max_b_frames + 1);

        s->rtp_mode   = 1;

        break;

    case AV_CODEC_ID_MJPEG:

        s->out_format = FMT_MJPEG;

        s->intra_only = 1; /* force intra only for jpeg */

        if (!CONFIG_MJPEG_ENCODER ||

            ff_mjpeg_encode_init(s) < 0)

            return -1;

        avctx->delay = 0;

        s->low_delay = 1;

        break;

    case AV_CODEC_ID_H261:

        if (!CONFIG_H261_ENCODER)

            return -1;

        if (ff_h261_get_picture_format(s->width, s->height) < 0) {

            av_log(avctx, AV_LOG_ERROR,

                   "The specified picture size of %dx%d is not valid for the "

                   "H.261 codec.\nValid sizes are 176x144, 352x288\n",

                    s->width, s->height);

            return -1;

        }

        s->out_format = FMT_H261;

        avctx->delay  = 0;

        s->low_delay  = 1;

        s->rtp_mode   = 0; /* Sliced encoding not supported */

        break;

    case AV_CODEC_ID_H263:

        if (!CONFIG_H263_ENCODER)

        return -1;

        if (ff_match_2uint16(ff_h263_format, FF_ARRAY_ELEMS(ff_h263_format),

                             s->width, s->height) == 8) {

            av_log(avctx, AV_LOG_INFO,

                   "The specified picture size of %dx%d is not valid for "

                   "the H.263 codec.\nValid sizes are 128x96, 176x144, "

                   "352x288, 704x576, and 1408x1152."

                   "Try H.263+.\n", s->width, s->height);

            return -1;

        }

        s->out_format = FMT_H263;

        avctx->delay  = 0;

        s->low_delay  = 1;

        break;

    case AV_CODEC_ID_H263P:

        s->out_format = FMT_H263;

        s->h263_plus  = 1;

        /* Fx */

        s->h263_aic        = (avctx->flags & AV_CODEC_FLAG_AC_PRED) ? 1 : 0;

        s->modified_quant  = s->h263_aic;

        s->loop_filter     = (avctx->flags & AV_CODEC_FLAG_LOOP_FILTER) ? 1 : 0;

        s->unrestricted_mv = s->obmc || s->loop_filter || s->umvplus;



        /* /Fx */

        /* These are just to be sure */

        avctx->delay = 0;

        s->low_delay = 1;

        break;

    case AV_CODEC_ID_FLV1:

        s->out_format      = FMT_H263;

        s->h263_flv        = 2; /* format = 1; 11-bit codes */

        s->unrestricted_mv = 1;

        s->rtp_mode  = 0; /* don't allow GOB */

        avctx->delay = 0;

        s->low_delay = 1;

        break;

    case AV_CODEC_ID_RV10:

        s->out_format = FMT_H263;

        avctx->delay  = 0;

        s->low_delay  = 1;

        break;

    case AV_CODEC_ID_RV20:

        s->out_format      = FMT_H263;

        avctx->delay       = 0;

        s->low_delay       = 1;

        s->modified_quant  = 1;

        s->h263_aic        = 1;

        s->h263_plus       = 1;

        s->loop_filter     = 1;

        s->unrestricted_mv = 0;

        break;

    case AV_CODEC_ID_MPEG4:

        s->out_format      = FMT_H263;

        s->h263_pred       = 1;

        s->unrestricted_mv = 1;

        s->low_delay       = s->max_b_frames ? 0 : 1;

        avctx->delay       = s->low_delay ? 0 : (s->max_b_frames + 1);

        break;

    case AV_CODEC_ID_MSMPEG4V2:

        s->out_format      = FMT_H263;

        s->h263_pred       = 1;

        s->unrestricted_mv = 1;

        s->msmpeg4_version = 2;

        avctx->delay       = 0;

        s->low_delay       = 1;

        break;

    case AV_CODEC_ID_MSMPEG4V3:

        s->out_format        = FMT_H263;

        s->h263_pred         = 1;

        s->unrestricted_mv   = 1;

        s->msmpeg4_version   = 3;

        s->flipflop_rounding = 1;

        avctx->delay         = 0;

        s->low_delay         = 1;

        break;

    case AV_CODEC_ID_WMV1:

        s->out_format        = FMT_H263;

        s->h263_pred         = 1;

        s->unrestricted_mv   = 1;

        s->msmpeg4_version   = 4;

        s->flipflop_rounding = 1;

        avctx->delay         = 0;

        s->low_delay         = 1;

        break;

    case AV_CODEC_ID_WMV2:

        s->out_format        = FMT_H263;

        s->h263_pred         = 1;

        s->unrestricted_mv   = 1;

        s->msmpeg4_version   = 5;

        s->flipflop_rounding = 1;

        avctx->delay         = 0;

        s->low_delay         = 1;

        break;

    default:

        return -1;

    }



    avctx->has_b_frames = !s->low_delay;



    s->encoding = 1;



    s->progressive_frame    =

    s->progressive_sequence = !(avctx->flags & (AV_CODEC_FLAG_INTERLACED_DCT |

                                                AV_CODEC_FLAG_INTERLACED_ME) ||

                                s->alternate_scan);



    /* init */

    ff_mpv_idct_init(s);

    if (ff_mpv_common_init(s) < 0)

        return -1;



    if (ARCH_X86)

        ff_mpv_encode_init_x86(s);



    ff_fdctdsp_init(&s->fdsp, avctx);

    ff_me_cmp_init(&s->mecc, avctx);

    ff_mpegvideoencdsp_init(&s->mpvencdsp, avctx);

    ff_pixblockdsp_init(&s->pdsp, avctx);

    ff_qpeldsp_init(&s->qdsp);



    if (s->msmpeg4_version) {

        FF_ALLOCZ_OR_GOTO(s->avctx, s->ac_stats,

                          2 * 2 * (MAX_LEVEL + 1) *

                          (MAX_RUN + 1) * 2 * sizeof(int), fail);

    }

    FF_ALLOCZ_OR_GOTO(s->avctx, s->avctx->stats_out, 256, fail);



    FF_ALLOCZ_OR_GOTO(s->avctx, s->q_intra_matrix,   64 * 32 * sizeof(int), fail);

    FF_ALLOCZ_OR_GOTO(s->avctx, s->q_inter_matrix,   64 * 32 * sizeof(int), fail);

    FF_ALLOCZ_OR_GOTO(s->avctx, s->q_intra_matrix16, 64 * 32 * 2 * sizeof(uint16_t), fail);

    FF_ALLOCZ_OR_GOTO(s->avctx, s->q_inter_matrix16, 64 * 32 * 2 * sizeof(uint16_t), fail);

    FF_ALLOCZ_OR_GOTO(s->avctx, s->input_picture,

                      MAX_PICTURE_COUNT * sizeof(Picture *), fail);

    FF_ALLOCZ_OR_GOTO(s->avctx, s->reordered_input_picture,

                      MAX_PICTURE_COUNT * sizeof(Picture *), fail);



    if (s->avctx->noise_reduction) {

        FF_ALLOCZ_OR_GOTO(s->avctx, s->dct_offset,

                          2 * 64 * sizeof(uint16_t), fail);

    }



    if (CONFIG_H263_ENCODER)

        ff_h263dsp_init(&s->h263dsp);

    if (!s->dct_quantize)

        s->dct_quantize = ff_dct_quantize_c;

    if (!s->denoise_dct)

        s->denoise_dct  = denoise_dct_c;

    s->fast_dct_quantize = s->dct_quantize;

    if (avctx->trellis)

        s->dct_quantize  = dct_quantize_trellis_c;



    if ((CONFIG_H263P_ENCODER || CONFIG_RV20_ENCODER) && s->modified_quant)

        s->chroma_qscale_table = ff_h263_chroma_qscale_table;



    s->quant_precision = 5;



    ff_set_cmp(&s->mecc, s->mecc.ildct_cmp,      s->avctx->ildct_cmp);

    ff_set_cmp(&s->mecc, s->mecc.frame_skip_cmp, s->avctx->frame_skip_cmp);



    if (CONFIG_H261_ENCODER && s->out_format == FMT_H261)

        ff_h261_encode_init(s);

    if (CONFIG_H263_ENCODER && s->out_format == FMT_H263)

        ff_h263_encode_init(s);

    if (CONFIG_MSMPEG4_ENCODER && s->msmpeg4_version)

        if ((ret = ff_msmpeg4_encode_init(s)) < 0)

            return ret;

    if ((CONFIG_MPEG1VIDEO_ENCODER || CONFIG_MPEG2VIDEO_ENCODER)

        && s->out_format == FMT_MPEG1)

        ff_mpeg1_encode_init(s);



    /* init q matrix */

    for (i = 0; i < 64; i++) {

        int j = s->idsp.idct_permutation[i];

        if (CONFIG_MPEG4_ENCODER && s->codec_id == AV_CODEC_ID_MPEG4 &&

            s->mpeg_quant) {

            s->intra_matrix[j] = ff_mpeg4_default_intra_matrix[i];

            s->inter_matrix[j] = ff_mpeg4_default_non_intra_matrix[i];

        } else if (s->out_format == FMT_H263 || s->out_format == FMT_H261) {

            s->intra_matrix[j] =

            s->inter_matrix[j] = ff_mpeg1_default_non_intra_matrix[i];

        } else {

            /* mpeg1/2 */

            s->intra_matrix[j] = ff_mpeg1_default_intra_matrix[i];

            s->inter_matrix[j] = ff_mpeg1_default_non_intra_matrix[i];

        }

        if (s->avctx->intra_matrix)

            s->intra_matrix[j] = s->avctx->intra_matrix[i];

        if (s->avctx->inter_matrix)

            s->inter_matrix[j] = s->avctx->inter_matrix[i];

    }



    /* precompute matrix */

    /* for mjpeg, we do include qscale in the matrix */

    if (s->out_format != FMT_MJPEG) {

        ff_convert_matrix(s, s->q_intra_matrix, s->q_intra_matrix16,

                          s->intra_matrix, s->intra_quant_bias, avctx->qmin,

                          31, 1);

        ff_convert_matrix(s, s->q_inter_matrix, s->q_inter_matrix16,

                          s->inter_matrix, s->inter_quant_bias, avctx->qmin,

                          31, 0);

    }



    if (ff_rate_control_init(s) < 0)

        return -1;



#if FF_API_ERROR_RATE

    FF_DISABLE_DEPRECATION_WARNINGS

    if (avctx->error_rate)

        s->error_rate = avctx->error_rate;

    FF_ENABLE_DEPRECATION_WARNINGS;

#endif



#if FF_API_NORMALIZE_AQP

    FF_DISABLE_DEPRECATION_WARNINGS

    if (avctx->flags & CODEC_FLAG_NORMALIZE_AQP)

        s->mpv_flags |= FF_MPV_FLAG_NAQ;

    FF_ENABLE_DEPRECATION_WARNINGS;

#endif



#if FF_API_MV0

    FF_DISABLE_DEPRECATION_WARNINGS

    if (avctx->flags & CODEC_FLAG_MV0)

        s->mpv_flags |= FF_MPV_FLAG_MV0;

    FF_ENABLE_DEPRECATION_WARNINGS

#endif



#if FF_API_MPV_OPT

    FF_DISABLE_DEPRECATION_WARNINGS

    if (avctx->rc_qsquish != 0.0)

        s->rc_qsquish = avctx->rc_qsquish;

    if (avctx->rc_qmod_amp != 0.0)

        s->rc_qmod_amp = avctx->rc_qmod_amp;

    if (avctx->rc_qmod_freq)

        s->rc_qmod_freq = avctx->rc_qmod_freq;

    if (avctx->rc_buffer_aggressivity != 1.0)

        s->rc_buffer_aggressivity = avctx->rc_buffer_aggressivity;

    if (avctx->rc_initial_cplx != 0.0)

        s->rc_initial_cplx = avctx->rc_initial_cplx;

    if (avctx->lmin)

        s->lmin = avctx->lmin;

    if (avctx->lmax)

        s->lmax = avctx->lmax;



    if (avctx->rc_eq) {

        av_freep(&s->rc_eq);

        s->rc_eq = av_strdup(avctx->rc_eq);

        if (!s->rc_eq)

            return AVERROR(ENOMEM);

    }

    FF_ENABLE_DEPRECATION_WARNINGS

#endif



    if (avctx->b_frame_strategy == 2) {

        for (i = 0; i < s->max_b_frames + 2; i++) {

            s->tmp_frames[i] = av_frame_alloc();

            if (!s->tmp_frames[i])

                return AVERROR(ENOMEM);



            s->tmp_frames[i]->format = AV_PIX_FMT_YUV420P;

            s->tmp_frames[i]->width  = s->width  >> avctx->brd_scale;

            s->tmp_frames[i]->height = s->height >> avctx->brd_scale;



            ret = av_frame_get_buffer(s->tmp_frames[i], 32);

            if (ret < 0)

                return ret;

        }

    }



    return 0;

fail:

    ff_mpv_encode_end(avctx);

    return AVERROR_UNKNOWN;

}
