static int dca_subframe_header(DCAContext *s, int base_channel, int block_index)

{

    /* Primary audio coding side information */

    int j, k;



    if (get_bits_left(&s->gb) < 0)

        return AVERROR_INVALIDDATA;



    if (!base_channel) {

        s->subsubframes[s->current_subframe]    = get_bits(&s->gb, 2) + 1;

        s->partial_samples[s->current_subframe] = get_bits(&s->gb, 3);

    }



    for (j = base_channel; j < s->prim_channels; j++) {

        for (k = 0; k < s->subband_activity[j]; k++)

            s->prediction_mode[j][k] = get_bits(&s->gb, 1);

    }



    /* Get prediction codebook */

    for (j = base_channel; j < s->prim_channels; j++) {

        for (k = 0; k < s->subband_activity[j]; k++) {

            if (s->prediction_mode[j][k] > 0) {

                /* (Prediction coefficient VQ address) */

                s->prediction_vq[j][k] = get_bits(&s->gb, 12);

            }

        }

    }



    /* Bit allocation index */

    for (j = base_channel; j < s->prim_channels; j++) {

        for (k = 0; k < s->vq_start_subband[j]; k++) {

            if (s->bitalloc_huffman[j] == 6)

                s->bitalloc[j][k] = get_bits(&s->gb, 5);

            else if (s->bitalloc_huffman[j] == 5)

                s->bitalloc[j][k] = get_bits(&s->gb, 4);

            else if (s->bitalloc_huffman[j] == 7) {

                av_log(s->avctx, AV_LOG_ERROR,

                       "Invalid bit allocation index\n");

                return AVERROR_INVALIDDATA;

            } else {

                s->bitalloc[j][k] =

                    get_bitalloc(&s->gb, &dca_bitalloc_index, s->bitalloc_huffman[j]);

            }



            if (s->bitalloc[j][k] > 26) {

                // av_log(s->avctx, AV_LOG_DEBUG, "bitalloc index [%i][%i] too big (%i)\n",

                //        j, k, s->bitalloc[j][k]);

                return AVERROR_INVALIDDATA;

            }

        }

    }



    /* Transition mode */

    for (j = base_channel; j < s->prim_channels; j++) {

        for (k = 0; k < s->subband_activity[j]; k++) {

            s->transition_mode[j][k] = 0;

            if (s->subsubframes[s->current_subframe] > 1 &&

                k < s->vq_start_subband[j] && s->bitalloc[j][k] > 0) {

                s->transition_mode[j][k] =

                    get_bitalloc(&s->gb, &dca_tmode, s->transient_huffman[j]);

            }

        }

    }



    if (get_bits_left(&s->gb) < 0)

        return AVERROR_INVALIDDATA;



    for (j = base_channel; j < s->prim_channels; j++) {

        const uint32_t *scale_table;

        int scale_sum, log_size;



        memset(s->scale_factor[j], 0,

               s->subband_activity[j] * sizeof(s->scale_factor[0][0][0]) * 2);



        if (s->scalefactor_huffman[j] == 6) {

            scale_table = scale_factor_quant7;

            log_size = 7;

        } else {

            scale_table = scale_factor_quant6;

            log_size = 6;

        }



        /* When huffman coded, only the difference is encoded */

        scale_sum = 0;



        for (k = 0; k < s->subband_activity[j]; k++) {

            if (k >= s->vq_start_subband[j] || s->bitalloc[j][k] > 0) {

                scale_sum = get_scale(&s->gb, s->scalefactor_huffman[j], scale_sum, log_size);

                s->scale_factor[j][k][0] = scale_table[scale_sum];

            }



            if (k < s->vq_start_subband[j] && s->transition_mode[j][k]) {

                /* Get second scale factor */

                scale_sum = get_scale(&s->gb, s->scalefactor_huffman[j], scale_sum, log_size);

                s->scale_factor[j][k][1] = scale_table[scale_sum];

            }

        }

    }



    /* Joint subband scale factor codebook select */

    for (j = base_channel; j < s->prim_channels; j++) {

        /* Transmitted only if joint subband coding enabled */

        if (s->joint_intensity[j] > 0)

            s->joint_huff[j] = get_bits(&s->gb, 3);

    }



    if (get_bits_left(&s->gb) < 0)

        return AVERROR_INVALIDDATA;



    /* Scale factors for joint subband coding */

    for (j = base_channel; j < s->prim_channels; j++) {

        int source_channel;



        /* Transmitted only if joint subband coding enabled */

        if (s->joint_intensity[j] > 0) {

            int scale = 0;

            source_channel = s->joint_intensity[j] - 1;



            /* When huffman coded, only the difference is encoded

             * (is this valid as well for joint scales ???) */



            for (k = s->subband_activity[j]; k < s->subband_activity[source_channel]; k++) {

                scale = get_scale(&s->gb, s->joint_huff[j], 64 /* bias */, 7);

                s->joint_scale_factor[j][k] = scale;    /*joint_scale_table[scale]; */

            }



            if (!(s->debug_flag & 0x02)) {

                av_log(s->avctx, AV_LOG_DEBUG,

                       "Joint stereo coding not supported\n");

                s->debug_flag |= 0x02;

            }

        }

    }



    /* Stereo downmix coefficients */

    if (!base_channel && s->prim_channels > 2) {

        if (s->downmix) {

            for (j = base_channel; j < s->prim_channels; j++) {

                s->downmix_coef[j][0] = get_bits(&s->gb, 7);

                s->downmix_coef[j][1] = get_bits(&s->gb, 7);

            }

        } else {

            int am = s->amode & DCA_CHANNEL_MASK;

            if (am >= FF_ARRAY_ELEMS(dca_default_coeffs)) {

                av_log(s->avctx, AV_LOG_ERROR,

                       "Invalid channel mode %d\n", am);

                return AVERROR_INVALIDDATA;

            }

            for (j = base_channel; j < s->prim_channels; j++) {

                s->downmix_coef[j][0] = dca_default_coeffs[am][j][0];

                s->downmix_coef[j][1] = dca_default_coeffs[am][j][1];

            }

        }

    }



    /* Dynamic range coefficient */

    if (!base_channel && s->dynrange)

        s->dynrange_coef = get_bits(&s->gb, 8);



    /* Side information CRC check word */

    if (s->crc_present) {

        get_bits(&s->gb, 16);

    }



    /*

     * Primary audio data arrays

     */



    /* VQ encoded high frequency subbands */

    for (j = base_channel; j < s->prim_channels; j++)

        for (k = s->vq_start_subband[j]; k < s->subband_activity[j]; k++)

            /* 1 vector -> 32 samples */

            s->high_freq_vq[j][k] = get_bits(&s->gb, 10);



    /* Low frequency effect data */

    if (!base_channel && s->lfe) {

        int quant7;

        /* LFE samples */

        int lfe_samples = 2 * s->lfe * (4 + block_index);

        int lfe_end_sample = 2 * s->lfe * (4 + block_index + s->subsubframes[s->current_subframe]);

        float lfe_scale;



        for (j = lfe_samples; j < lfe_end_sample; j++) {

            /* Signed 8 bits int */

            s->lfe_data[j] = get_sbits(&s->gb, 8);

        }



        /* Scale factor index */

        quant7 = get_bits(&s->gb, 8);

        if (quant7 > 127) {

            av_log_ask_for_sample(s->avctx, "LFEScaleIndex larger than 127\n");

            return AVERROR_INVALIDDATA;

        }

        s->lfe_scale_factor = scale_factor_quant7[quant7];



        /* Quantization step size * scale factor */

        lfe_scale = 0.035 * s->lfe_scale_factor;



        for (j = lfe_samples; j < lfe_end_sample; j++)

            s->lfe_data[j] *= lfe_scale;

    }



#ifdef TRACE

    av_log(s->avctx, AV_LOG_DEBUG, "subsubframes: %i\n",

           s->subsubframes[s->current_subframe]);

    av_log(s->avctx, AV_LOG_DEBUG, "partial samples: %i\n",

           s->partial_samples[s->current_subframe]);



    for (j = base_channel; j < s->prim_channels; j++) {

        av_log(s->avctx, AV_LOG_DEBUG, "prediction mode:");

        for (k = 0; k < s->subband_activity[j]; k++)

            av_log(s->avctx, AV_LOG_DEBUG, " %i", s->prediction_mode[j][k]);

        av_log(s->avctx, AV_LOG_DEBUG, "\n");

    }

    for (j = base_channel; j < s->prim_channels; j++) {

        for (k = 0; k < s->subband_activity[j]; k++)

            av_log(s->avctx, AV_LOG_DEBUG,

                   "prediction coefs: %f, %f, %f, %f\n",

                   (float) adpcm_vb[s->prediction_vq[j][k]][0] / 8192,

                   (float) adpcm_vb[s->prediction_vq[j][k]][1] / 8192,

                   (float) adpcm_vb[s->prediction_vq[j][k]][2] / 8192,

                   (float) adpcm_vb[s->prediction_vq[j][k]][3] / 8192);

    }

    for (j = base_channel; j < s->prim_channels; j++) {

        av_log(s->avctx, AV_LOG_DEBUG, "bitalloc index: ");

        for (k = 0; k < s->vq_start_subband[j]; k++)

            av_log(s->avctx, AV_LOG_DEBUG, "%2.2i ", s->bitalloc[j][k]);

        av_log(s->avctx, AV_LOG_DEBUG, "\n");

    }

    for (j = base_channel; j < s->prim_channels; j++) {

        av_log(s->avctx, AV_LOG_DEBUG, "Transition mode:");

        for (k = 0; k < s->subband_activity[j]; k++)

            av_log(s->avctx, AV_LOG_DEBUG, " %i", s->transition_mode[j][k]);

        av_log(s->avctx, AV_LOG_DEBUG, "\n");

    }

    for (j = base_channel; j < s->prim_channels; j++) {

        av_log(s->avctx, AV_LOG_DEBUG, "Scale factor:");

        for (k = 0; k < s->subband_activity[j]; k++) {

            if (k >= s->vq_start_subband[j] || s->bitalloc[j][k] > 0)

                av_log(s->avctx, AV_LOG_DEBUG, " %i", s->scale_factor[j][k][0]);

            if (k < s->vq_start_subband[j] && s->transition_mode[j][k])

                av_log(s->avctx, AV_LOG_DEBUG, " %i(t)", s->scale_factor[j][k][1]);

        }

        av_log(s->avctx, AV_LOG_DEBUG, "\n");

    }

    for (j = base_channel; j < s->prim_channels; j++) {

        if (s->joint_intensity[j] > 0) {

            int source_channel = s->joint_intensity[j] - 1;

            av_log(s->avctx, AV_LOG_DEBUG, "Joint scale factor index:\n");

            for (k = s->subband_activity[j]; k < s->subband_activity[source_channel]; k++)

                av_log(s->avctx, AV_LOG_DEBUG, " %i", s->joint_scale_factor[j][k]);

            av_log(s->avctx, AV_LOG_DEBUG, "\n");

        }

    }

    if (!base_channel && s->prim_channels > 2 && s->downmix) {

        av_log(s->avctx, AV_LOG_DEBUG, "Downmix coeffs:\n");

        for (j = 0; j < s->prim_channels; j++) {

            av_log(s->avctx, AV_LOG_DEBUG, "Channel 0, %d = %f\n", j,

                   dca_downmix_coeffs[s->downmix_coef[j][0]]);

            av_log(s->avctx, AV_LOG_DEBUG, "Channel 1, %d = %f\n", j,

                   dca_downmix_coeffs[s->downmix_coef[j][1]]);

        }

        av_log(s->avctx, AV_LOG_DEBUG, "\n");

    }

    for (j = base_channel; j < s->prim_channels; j++)

        for (k = s->vq_start_subband[j]; k < s->subband_activity[j]; k++)

            av_log(s->avctx, AV_LOG_DEBUG, "VQ index: %i\n", s->high_freq_vq[j][k]);

    if (!base_channel && s->lfe) {

        int lfe_samples = 2 * s->lfe * (4 + block_index);

        int lfe_end_sample = 2 * s->lfe * (4 + block_index + s->subsubframes[s->current_subframe]);



        av_log(s->avctx, AV_LOG_DEBUG, "LFE samples:\n");

        for (j = lfe_samples; j < lfe_end_sample; j++)

            av_log(s->avctx, AV_LOG_DEBUG, " %f", s->lfe_data[j]);

        av_log(s->avctx, AV_LOG_DEBUG, "\n");

    }

#endif



    return 0;

}
