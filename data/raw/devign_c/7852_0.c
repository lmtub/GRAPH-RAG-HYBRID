static av_cold int encode_init(AVCodecContext *avctx)

{

    FFV1Context *s = avctx->priv_data;

    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(avctx->pix_fmt);

    int i, j, k, m, ret;



    if ((ret = ff_ffv1_common_init(avctx)) < 0)

        return ret;



    s->version = 0;



    if ((avctx->flags & (AV_CODEC_FLAG_PASS1 | AV_CODEC_FLAG_PASS2)) ||

        avctx->slices > 1)

        s->version = FFMAX(s->version, 2);



    // Unspecified level & slices, we choose version 1.2+ to ensure multithreaded decodability

    if (avctx->slices == 0 && avctx->level < 0 && avctx->width * avctx->height > 720*576)

        s->version = FFMAX(s->version, 2);



    if (avctx->level <= 0 && s->version == 2) {

        s->version = 3;

    }

    if (avctx->level >= 0 && avctx->level <= 4) {

        if (avctx->level < s->version) {

            av_log(avctx, AV_LOG_ERROR, "Version %d needed for requested features but %d requested\n", s->version, avctx->level);

            return AVERROR(EINVAL);

        }

        s->version = avctx->level;

    }



    if (s->ec < 0) {

        s->ec = (s->version >= 3);

    }



    if ((s->version == 2 || s->version>3) && avctx->strict_std_compliance > FF_COMPLIANCE_EXPERIMENTAL) {

        av_log(avctx, AV_LOG_ERROR, "Version 2 needed for requested features but version 2 is experimental and not enabled\n");

        return AVERROR_INVALIDDATA;

    }



#if FF_API_CODER_TYPE

FF_DISABLE_DEPRECATION_WARNINGS

    if (avctx->coder_type != -1)

        s->ac = avctx->coder_type > 0 ? AC_RANGE_CUSTOM_TAB : AC_GOLOMB_RICE;

    else

FF_ENABLE_DEPRECATION_WARNINGS

#endif

    if (s->ac == 1) // Compatbility with common command line usage

        s->ac = AC_RANGE_CUSTOM_TAB;

    else if (s->ac == AC_RANGE_DEFAULT_TAB_FORCE)

        s->ac = AC_RANGE_DEFAULT_TAB;



    s->plane_count = 3;

    switch(avctx->pix_fmt) {

    case AV_PIX_FMT_YUV444P9:

    case AV_PIX_FMT_YUV422P9:

    case AV_PIX_FMT_YUV420P9:

    case AV_PIX_FMT_YUVA444P9:

    case AV_PIX_FMT_YUVA422P9:

    case AV_PIX_FMT_YUVA420P9:

        if (!avctx->bits_per_raw_sample)

            s->bits_per_raw_sample = 9;

    case AV_PIX_FMT_GRAY10:

    case AV_PIX_FMT_YUV444P10:

    case AV_PIX_FMT_YUV420P10:

    case AV_PIX_FMT_YUV422P10:

    case AV_PIX_FMT_YUVA444P10:

    case AV_PIX_FMT_YUVA422P10:

    case AV_PIX_FMT_YUVA420P10:

        if (!avctx->bits_per_raw_sample && !s->bits_per_raw_sample)

            s->bits_per_raw_sample = 10;

    case AV_PIX_FMT_GRAY12:

    case AV_PIX_FMT_YUV444P12:

    case AV_PIX_FMT_YUV420P12:

    case AV_PIX_FMT_YUV422P12:

        s->packed_at_lsb = 1;

        if (!avctx->bits_per_raw_sample && !s->bits_per_raw_sample)

            s->bits_per_raw_sample = 12;

    case AV_PIX_FMT_GRAY16:

    case AV_PIX_FMT_YUV444P16:

    case AV_PIX_FMT_YUV422P16:

    case AV_PIX_FMT_YUV420P16:

    case AV_PIX_FMT_YUVA444P16:

    case AV_PIX_FMT_YUVA422P16:

    case AV_PIX_FMT_YUVA420P16:

        if (!avctx->bits_per_raw_sample && !s->bits_per_raw_sample) {

            s->bits_per_raw_sample = 16;

        } else if (!s->bits_per_raw_sample) {

            s->bits_per_raw_sample = avctx->bits_per_raw_sample;

        }

        if (s->bits_per_raw_sample <= 8) {

            av_log(avctx, AV_LOG_ERROR, "bits_per_raw_sample invalid\n");

            return AVERROR_INVALIDDATA;

        }

        s->version = FFMAX(s->version, 1);

    case AV_PIX_FMT_GRAY8:

    case AV_PIX_FMT_YA8:

    case AV_PIX_FMT_YUV444P:

    case AV_PIX_FMT_YUV440P:

    case AV_PIX_FMT_YUV422P:

    case AV_PIX_FMT_YUV420P:

    case AV_PIX_FMT_YUV411P:

    case AV_PIX_FMT_YUV410P:

    case AV_PIX_FMT_YUVA444P:

    case AV_PIX_FMT_YUVA422P:

    case AV_PIX_FMT_YUVA420P:

        s->chroma_planes = desc->nb_components < 3 ? 0 : 1;

        s->colorspace = 0;

        s->transparency = desc->nb_components == 4 || desc->nb_components == 2;

        if (!avctx->bits_per_raw_sample && !s->bits_per_raw_sample)

            s->bits_per_raw_sample = 8;

        else if (!s->bits_per_raw_sample)

            s->bits_per_raw_sample = 8;

        break;

    case AV_PIX_FMT_RGB32:

        s->colorspace = 1;

        s->transparency = 1;

        s->chroma_planes = 1;

        s->bits_per_raw_sample = 8;

        break;

    case AV_PIX_FMT_RGB48:

        s->colorspace = 1;

        s->chroma_planes = 1;

        s->bits_per_raw_sample = 16;

        s->use32bit = 1;

        s->version = FFMAX(s->version, 1);

        if (avctx->strict_std_compliance > FF_COMPLIANCE_EXPERIMENTAL) {

            av_log(avctx, AV_LOG_ERROR, "16bit RGB is experimental and under development, only use it for experiments\n");

            return AVERROR_INVALIDDATA;

        }

        break;

    case AV_PIX_FMT_0RGB32:

        s->colorspace = 1;

        s->chroma_planes = 1;

        s->bits_per_raw_sample = 8;

        break;

    case AV_PIX_FMT_GBRP9:

        if (!avctx->bits_per_raw_sample)

            s->bits_per_raw_sample = 9;

    case AV_PIX_FMT_GBRP10:

        if (!avctx->bits_per_raw_sample && !s->bits_per_raw_sample)

            s->bits_per_raw_sample = 10;

    case AV_PIX_FMT_GBRP12:

        if (!avctx->bits_per_raw_sample && !s->bits_per_raw_sample)

            s->bits_per_raw_sample = 12;

    case AV_PIX_FMT_GBRP14:

        if (!avctx->bits_per_raw_sample && !s->bits_per_raw_sample)

            s->bits_per_raw_sample = 14;

    case AV_PIX_FMT_GBRP16:

        if (!avctx->bits_per_raw_sample && !s->bits_per_raw_sample)

            s->bits_per_raw_sample = 16;

        else if (!s->bits_per_raw_sample)

            s->bits_per_raw_sample = avctx->bits_per_raw_sample;

        s->colorspace = 1;

        s->chroma_planes = 1;

        if (s->bits_per_raw_sample >= 16) {

            s->use32bit = 1;

            if (avctx->strict_std_compliance > FF_COMPLIANCE_EXPERIMENTAL) {

                av_log(avctx, AV_LOG_ERROR, "16bit RGB is experimental and under development, only use it for experiments\n");

                return AVERROR_INVALIDDATA;

            }

        }

        s->version = FFMAX(s->version, 1);

        break;

    default:

        av_log(avctx, AV_LOG_ERROR, "format not supported\n");

        return AVERROR(ENOSYS);

    }

    av_assert0(s->bits_per_raw_sample >= 8);



    if (s->bits_per_raw_sample > 8) {

        if (s->ac == AC_GOLOMB_RICE) {

            av_log(avctx, AV_LOG_INFO,

                    "bits_per_raw_sample > 8, forcing range coder\n");

            s->ac = AC_RANGE_CUSTOM_TAB;

        }

    }

    if (s->transparency) {

        av_log(avctx, AV_LOG_WARNING, "Storing alpha plane, this will require a recent FFV1 decoder to playback!\n");

    }

#if FF_API_PRIVATE_OPT

FF_DISABLE_DEPRECATION_WARNINGS

    if (avctx->context_model)

        s->context_model = avctx->context_model;

    if (avctx->context_model > 1U) {

        av_log(avctx, AV_LOG_ERROR, "Invalid context model %d, valid values are 0 and 1\n", avctx->context_model);

        return AVERROR(EINVAL);

    }

FF_ENABLE_DEPRECATION_WARNINGS

#endif



    if (s->ac == AC_RANGE_CUSTOM_TAB) {

        for (i = 1; i < 256; i++)

            s->state_transition[i] = ver2_state[i];

    } else {

        RangeCoder c;

        ff_build_rac_states(&c, 0.05 * (1LL << 32), 256 - 8);

        for (i = 1; i < 256; i++)

            s->state_transition[i] = c.one_state[i];

    }



    for (i = 0; i < 256; i++) {

        s->quant_table_count = 2;

        if (s->bits_per_raw_sample <= 8) {

            s->quant_tables[0][0][i]=           quant11[i];

            s->quant_tables[0][1][i]=        11*quant11[i];

            s->quant_tables[0][2][i]=     11*11*quant11[i];

            s->quant_tables[1][0][i]=           quant11[i];

            s->quant_tables[1][1][i]=        11*quant11[i];

            s->quant_tables[1][2][i]=     11*11*quant5 [i];

            s->quant_tables[1][3][i]=   5*11*11*quant5 [i];

            s->quant_tables[1][4][i]= 5*5*11*11*quant5 [i];

        } else {

            s->quant_tables[0][0][i]=           quant9_10bit[i];

            s->quant_tables[0][1][i]=        11*quant9_10bit[i];

            s->quant_tables[0][2][i]=     11*11*quant9_10bit[i];

            s->quant_tables[1][0][i]=           quant9_10bit[i];

            s->quant_tables[1][1][i]=        11*quant9_10bit[i];

            s->quant_tables[1][2][i]=     11*11*quant5_10bit[i];

            s->quant_tables[1][3][i]=   5*11*11*quant5_10bit[i];

            s->quant_tables[1][4][i]= 5*5*11*11*quant5_10bit[i];

        }

    }

    s->context_count[0] = (11 * 11 * 11        + 1) / 2;

    s->context_count[1] = (11 * 11 * 5 * 5 * 5 + 1) / 2;

    memcpy(s->quant_table, s->quant_tables[s->context_model],

           sizeof(s->quant_table));



    for (i = 0; i < s->plane_count; i++) {

        PlaneContext *const p = &s->plane[i];



        memcpy(p->quant_table, s->quant_table, sizeof(p->quant_table));

        p->quant_table_index = s->context_model;

        p->context_count     = s->context_count[p->quant_table_index];

    }



    if ((ret = ff_ffv1_allocate_initial_states(s)) < 0)

        return ret;



#if FF_API_CODED_FRAME

FF_DISABLE_DEPRECATION_WARNINGS

    avctx->coded_frame->pict_type = AV_PICTURE_TYPE_I;

FF_ENABLE_DEPRECATION_WARNINGS

#endif



    if (!s->transparency)

        s->plane_count = 2;

    if (!s->chroma_planes && s->version > 3)

        s->plane_count--;



    avcodec_get_chroma_sub_sample(avctx->pix_fmt, &s->chroma_h_shift, &s->chroma_v_shift);

    s->picture_number = 0;



    if (avctx->flags & (AV_CODEC_FLAG_PASS1 | AV_CODEC_FLAG_PASS2)) {

        for (i = 0; i < s->quant_table_count; i++) {

            s->rc_stat2[i] = av_mallocz(s->context_count[i] *

                                        sizeof(*s->rc_stat2[i]));

            if (!s->rc_stat2[i])

                return AVERROR(ENOMEM);

        }

    }

    if (avctx->stats_in) {

        char *p = avctx->stats_in;

        uint8_t (*best_state)[256] = av_malloc_array(256, 256);

        int gob_count = 0;

        char *next;

        if (!best_state)

            return AVERROR(ENOMEM);



        av_assert0(s->version >= 2);



        for (;;) {

            for (j = 0; j < 256; j++)

                for (i = 0; i < 2; i++) {

                    s->rc_stat[j][i] = strtol(p, &next, 0);

                    if (next == p) {

                        av_log(avctx, AV_LOG_ERROR,

                               "2Pass file invalid at %d %d [%s]\n", j, i, p);

                        av_freep(&best_state);

                        return AVERROR_INVALIDDATA;

                    }

                    p = next;

                }

            for (i = 0; i < s->quant_table_count; i++)

                for (j = 0; j < s->context_count[i]; j++) {

                    for (k = 0; k < 32; k++)

                        for (m = 0; m < 2; m++) {

                            s->rc_stat2[i][j][k][m] = strtol(p, &next, 0);

                            if (next == p) {

                                av_log(avctx, AV_LOG_ERROR,

                                       "2Pass file invalid at %d %d %d %d [%s]\n",

                                       i, j, k, m, p);

                                av_freep(&best_state);

                                return AVERROR_INVALIDDATA;

                            }

                            p = next;

                        }

                }

            gob_count = strtol(p, &next, 0);

            if (next == p || gob_count <= 0) {

                av_log(avctx, AV_LOG_ERROR, "2Pass file invalid\n");

                av_freep(&best_state);

                return AVERROR_INVALIDDATA;

            }

            p = next;

            while (*p == '\n' || *p == ' ')

                p++;

            if (p[0] == 0)

                break;

        }

        if (s->ac == AC_RANGE_CUSTOM_TAB)

            sort_stt(s, s->state_transition);



        find_best_state(best_state, s->state_transition);



        for (i = 0; i < s->quant_table_count; i++) {

            for (k = 0; k < 32; k++) {

                double a=0, b=0;

                int jp = 0;

                for (j = 0; j < s->context_count[i]; j++) {

                    double p = 128;

                    if (s->rc_stat2[i][j][k][0] + s->rc_stat2[i][j][k][1] > 200 && j || a+b > 200) {

                        if (a+b)

                            p = 256.0 * b / (a + b);

                        s->initial_states[i][jp][k] =

                            best_state[av_clip(round(p), 1, 255)][av_clip_uint8((a + b) / gob_count)];

                        for(jp++; jp<j; jp++)

                            s->initial_states[i][jp][k] = s->initial_states[i][jp-1][k];

                        a=b=0;

                    }

                    a += s->rc_stat2[i][j][k][0];

                    b += s->rc_stat2[i][j][k][1];

                    if (a+b) {

                        p = 256.0 * b / (a + b);

                    }

                    s->initial_states[i][j][k] =

                        best_state[av_clip(round(p), 1, 255)][av_clip_uint8((a + b) / gob_count)];

                }

            }

        }

        av_freep(&best_state);

    }



    if (s->version > 1) {

        int plane_count = 1 + 2*s->chroma_planes + s->transparency;

        s->num_v_slices = (avctx->width > 352 || avctx->height > 288 || !avctx->slices) ? 2 : 1;



        if (avctx->height < 5)

            s->num_v_slices = 1;



        for (; s->num_v_slices < 32; s->num_v_slices++) {

            for (s->num_h_slices = s->num_v_slices; s->num_h_slices < 2*s->num_v_slices; s->num_h_slices++) {

                int maxw = (avctx->width  + s->num_h_slices - 1) / s->num_h_slices;

                int maxh = (avctx->height + s->num_v_slices - 1) / s->num_v_slices;

                if (s->num_h_slices > avctx->width || s->num_v_slices > avctx->height)

                    continue;

                if (maxw * maxh * (int64_t)(s->bits_per_raw_sample+1) * plane_count > 8<<24)

                    continue;

                if (avctx->slices == s->num_h_slices * s->num_v_slices && avctx->slices <= MAX_SLICES || !avctx->slices)

                    goto slices_ok;

            }

        }

        av_log(avctx, AV_LOG_ERROR,

               "Unsupported number %d of slices requested, please specify a "

               "supported number with -slices (ex:4,6,9,12,16, ...)\n",

               avctx->slices);

        return AVERROR(ENOSYS);

slices_ok:

        if ((ret = write_extradata(s)) < 0)

            return ret;

    }



    if ((ret = ff_ffv1_init_slice_contexts(s)) < 0)

        return ret;

    s->slice_count = s->max_slice_count;

    if ((ret = ff_ffv1_init_slices_state(s)) < 0)

        return ret;



#define STATS_OUT_SIZE 1024 * 1024 * 6

    if (avctx->flags & AV_CODEC_FLAG_PASS1) {

        avctx->stats_out = av_mallocz(STATS_OUT_SIZE);

        if (!avctx->stats_out)

            return AVERROR(ENOMEM);

        for (i = 0; i < s->quant_table_count; i++)

            for (j = 0; j < s->max_slice_count; j++) {

                FFV1Context *sf = s->slice_context[j];

                av_assert0(!sf->rc_stat2[i]);

                sf->rc_stat2[i] = av_mallocz(s->context_count[i] *

                                             sizeof(*sf->rc_stat2[i]));

                if (!sf->rc_stat2[i])

                    return AVERROR(ENOMEM);

            }

    }



    return 0;

}
