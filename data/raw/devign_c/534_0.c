static av_cold int ffv1_encode_init(AVCodecContext *avctx)

{

    FFV1Context *s = avctx->priv_data;

    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(avctx->pix_fmt);

    int i, j, k, m, ret;



    ffv1_common_init(avctx);



    s->version = 0;



    if ((avctx->flags & (CODEC_FLAG_PASS1 | CODEC_FLAG_PASS2)) ||

        avctx->slices > 1)

        s->version = FFMAX(s->version, 2);



    if (avctx->level == 3) {

        s->version = 3;

    }



    if (s->ec < 0) {

        s->ec = (s->version >= 3);

    }



    if (s->version >= 2 &&

        avctx->strict_std_compliance > FF_COMPLIANCE_EXPERIMENTAL) {

        av_log(avctx, AV_LOG_ERROR,

               "Version %d requested, please set -strict experimental in "

               "order to enable it\n",

               s->version);

        return AVERROR(ENOSYS);

    }



    s->ac = avctx->coder_type > 0 ? 2 : 0;



    s->plane_count = 3;

    switch (avctx->pix_fmt) {

    case AV_PIX_FMT_YUV444P9:

    case AV_PIX_FMT_YUV422P9:

    case AV_PIX_FMT_YUV420P9:

        if (!avctx->bits_per_raw_sample)

            s->bits_per_raw_sample = 9;

    case AV_PIX_FMT_YUV444P10:

    case AV_PIX_FMT_YUV420P10:

    case AV_PIX_FMT_YUV422P10:

        s->packed_at_lsb = 1;

        if (!avctx->bits_per_raw_sample && !s->bits_per_raw_sample)

            s->bits_per_raw_sample = 10;

    case AV_PIX_FMT_GRAY16:

    case AV_PIX_FMT_YUV444P16:

    case AV_PIX_FMT_YUV422P16:

    case AV_PIX_FMT_YUV420P16:

        if (!avctx->bits_per_raw_sample && !s->bits_per_raw_sample) {

            s->bits_per_raw_sample = 16;

        } else if (!s->bits_per_raw_sample) {

            s->bits_per_raw_sample = avctx->bits_per_raw_sample;

        }

        if (s->bits_per_raw_sample <= 8) {

            av_log(avctx, AV_LOG_ERROR, "bits_per_raw_sample invalid\n");

            return AVERROR_INVALIDDATA;

        }

        if (!s->ac && avctx->coder_type == -1) {

            av_log(avctx, AV_LOG_INFO,

                   "bits_per_raw_sample > 8, forcing coder 1\n");

            s->ac = 2;

        }

        if (!s->ac) {

            av_log(

                avctx, AV_LOG_ERROR,

                "bits_per_raw_sample of more than 8 needs -coder 1 currently\n");

            return AVERROR_INVALIDDATA;

        }

        s->version = FFMAX(s->version, 1);

    case AV_PIX_FMT_GRAY8:

    case AV_PIX_FMT_YUV444P:

    case AV_PIX_FMT_YUV440P:

    case AV_PIX_FMT_YUV422P:

    case AV_PIX_FMT_YUV420P:

    case AV_PIX_FMT_YUV411P:

    case AV_PIX_FMT_YUV410P:

        s->chroma_planes = desc->nb_components < 3 ? 0 : 1;

        s->colorspace    = 0;

        break;

    case AV_PIX_FMT_YUVA444P:

    case AV_PIX_FMT_YUVA422P:

    case AV_PIX_FMT_YUVA420P:

        s->chroma_planes = 1;

        s->colorspace    = 0;

        s->transparency  = 1;

        break;

    case AV_PIX_FMT_RGB32:

        s->colorspace   = 1;

        s->transparency = 1;

        break;

    case AV_PIX_FMT_GBRP9:

        if (!avctx->bits_per_raw_sample)

            s->bits_per_raw_sample = 9;

    case AV_PIX_FMT_GBRP10:

        if (!avctx->bits_per_raw_sample && !s->bits_per_raw_sample)

            s->bits_per_raw_sample = 10;

    case AV_PIX_FMT_GBRP16:

        if (!avctx->bits_per_raw_sample && !s->bits_per_raw_sample)

            s->bits_per_raw_sample = 16;

        else if (!s->bits_per_raw_sample)

            s->bits_per_raw_sample = avctx->bits_per_raw_sample;

        s->colorspace    = 1;

        s->chroma_planes = 1;

        s->version       = FFMAX(s->version, 1);

        break;

    default:

        av_log(avctx, AV_LOG_ERROR, "format not supported\n");

        return AVERROR_INVALIDDATA;

    }

    if (s->transparency) {

        av_log(

            avctx, AV_LOG_WARNING,

            "Storing alpha plane, this will require a recent FFV1 decoder to playback!\n");

    }

    if (avctx->context_model > 1U) {

        av_log(avctx, AV_LOG_ERROR,

               "Invalid context model %d, valid values are 0 and 1\n",

               avctx->context_model);

        return AVERROR(EINVAL);

    }



    if (s->ac > 1)

        for (i = 1; i < 256; i++)

            s->state_transition[i] = ffv1_ver2_state[i];



    for (i = 0; i < 256; i++) {

        s->quant_table_count = 2;

        if (s->bits_per_raw_sample <= 8) {

            s->quant_tables[0][0][i] = ffv1_quant11[i];

            s->quant_tables[0][1][i] = ffv1_quant11[i] * 11;

            s->quant_tables[0][2][i] = ffv1_quant11[i] * 11 * 11;

            s->quant_tables[1][0][i] = ffv1_quant11[i];

            s->quant_tables[1][1][i] = ffv1_quant11[i] * 11;

            s->quant_tables[1][2][i] = ffv1_quant5[i]  * 11 * 11;

            s->quant_tables[1][3][i] = ffv1_quant5[i]  *  5 * 11 * 11;

            s->quant_tables[1][4][i] = ffv1_quant5[i]  *  5 *  5 * 11 * 11;

        } else {

            s->quant_tables[0][0][i] = ffv1_quant9_10bit[i];

            s->quant_tables[0][1][i] = ffv1_quant9_10bit[i] * 11;

            s->quant_tables[0][2][i] = ffv1_quant9_10bit[i] * 11 * 11;

            s->quant_tables[1][0][i] = ffv1_quant9_10bit[i];

            s->quant_tables[1][1][i] = ffv1_quant9_10bit[i] * 11;

            s->quant_tables[1][2][i] = ffv1_quant5_10bit[i] * 11 * 11;

            s->quant_tables[1][3][i] = ffv1_quant5_10bit[i] *  5 * 11 * 11;

            s->quant_tables[1][4][i] = ffv1_quant5_10bit[i] *  5 *  5 * 11 * 11;

        }

    }

    s->context_count[0] = (11 * 11 * 11        + 1) / 2;

    s->context_count[1] = (11 * 11 * 5 * 5 * 5 + 1) / 2;

    memcpy(s->quant_table, s->quant_tables[avctx->context_model],

           sizeof(s->quant_table));



    for (i = 0; i < s->plane_count; i++) {

        PlaneContext *const p = &s->plane[i];



        memcpy(p->quant_table, s->quant_table, sizeof(p->quant_table));

        p->quant_table_index = avctx->context_model;

        p->context_count     = s->context_count[p->quant_table_index];

    }



    if ((ret = ffv1_allocate_initial_states(s)) < 0)

        return ret;



    avctx->coded_frame = av_frame_alloc();

    if (!avctx->coded_frame)

        return AVERROR(ENOMEM);



    avctx->coded_frame->pict_type = AV_PICTURE_TYPE_I;



    if (!s->transparency)

        s->plane_count = 2;



    av_pix_fmt_get_chroma_sub_sample(avctx->pix_fmt, &s->chroma_h_shift,

                                     &s->chroma_v_shift);



    s->picture_number = 0;



    if (avctx->flags & (CODEC_FLAG_PASS1 | CODEC_FLAG_PASS2)) {

        for (i = 0; i < s->quant_table_count; i++) {

            s->rc_stat2[i] = av_mallocz(s->context_count[i] *

                                        sizeof(*s->rc_stat2[i]));

            if (!s->rc_stat2[i])

                return AVERROR(ENOMEM);

        }

    }

    if (avctx->stats_in) {

        char *p = avctx->stats_in;

        uint8_t best_state[256][256];

        int gob_count = 0;

        char *next;



        av_assert0(s->version >= 2);



        for (;; ) {

            for (j = 0; j < 256; j++)

                for (i = 0; i < 2; i++) {

                    s->rc_stat[j][i] = strtol(p, &next, 0);

                    if (next == p) {

                        av_log(avctx, AV_LOG_ERROR,

                               "2Pass file invalid at %d %d [%s]\n", j, i, p);

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

                                return AVERROR_INVALIDDATA;

                            }

                            p = next;

                        }

                }

            gob_count = strtol(p, &next, 0);

            if (next == p || gob_count <= 0) {

                av_log(avctx, AV_LOG_ERROR, "2Pass file invalid\n");

                return AVERROR_INVALIDDATA;

            }

            p = next;

            while (*p == '\n' || *p == ' ')

                p++;

            if (p[0] == 0)

                break;

        }

        sort_stt(s, s->state_transition);



        find_best_state(best_state, s->state_transition);



        for (i = 0; i < s->quant_table_count; i++) {

            for (j = 0; j < s->context_count[i]; j++)

                for (k = 0; k < 32; k++) {

                    double p = 128;

                    if (s->rc_stat2[i][j][k][0] + s->rc_stat2[i][j][k][1]) {

                        p = 256.0 * s->rc_stat2[i][j][k][1] /

                            (s->rc_stat2[i][j][k][0] + s->rc_stat2[i][j][k][1]);

                    }

                    s->initial_states[i][j][k] =

                        best_state[av_clip(round(p), 1, 255)][av_clip((s->rc_stat2[i][j][k][0] +

                                                                       s->rc_stat2[i][j][k][1]) /

                                                                      gob_count, 0, 255)];

                }

        }

    }



    if (s->version > 1) {

        for (s->num_v_slices = 2; s->num_v_slices < 9; s->num_v_slices++)

            for (s->num_h_slices = s->num_v_slices;

                 s->num_h_slices < 2 * s->num_v_slices; s->num_h_slices++)

                if (avctx->slices == s->num_h_slices * s->num_v_slices &&

                    avctx->slices <= 64 || !avctx->slices)

                    goto slices_ok;

        av_log(avctx, AV_LOG_ERROR,

               "Unsupported number %d of slices requested, please specify a "

               "supported number with -slices (ex:4,6,9,12,16, ...)\n",

               avctx->slices);

        return AVERROR(ENOSYS);

slices_ok:

        write_extradata(s);

    }



    if ((ret = ffv1_init_slice_contexts(s)) < 0)

        return ret;

    if ((ret = init_slices_state(s)) < 0)

        return ret;



#define STATS_OUT_SIZE 1024 * 1024 * 6

    if (avctx->flags & CODEC_FLAG_PASS1) {

        avctx->stats_out = av_mallocz(STATS_OUT_SIZE);

        for (i = 0; i < s->quant_table_count; i++)

            for (j = 0; j < s->slice_count; j++) {

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
