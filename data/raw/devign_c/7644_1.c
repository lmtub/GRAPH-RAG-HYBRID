static int decode_spectrum_and_dequant(AACContext * ac, float coef[1024], GetBitContext * gb, float sf[120],

        int pulse_present, const Pulse * pulse, const IndividualChannelStream * ics, enum BandType band_type[120]) {

    int i, k, g, idx = 0;

    const int c = 1024/ics->num_windows;

    const uint16_t * offsets = ics->swb_offset;

    float *coef_base = coef;



    for (g = 0; g < ics->num_windows; g++)

        memset(coef + g * 128 + offsets[ics->max_sfb], 0, sizeof(float)*(c - offsets[ics->max_sfb]));



    for (g = 0; g < ics->num_window_groups; g++) {

        for (i = 0; i < ics->max_sfb; i++, idx++) {

            const int cur_band_type = band_type[idx];

            const int dim = cur_band_type >= FIRST_PAIR_BT ? 2 : 4;

            const int is_cb_unsigned = IS_CODEBOOK_UNSIGNED(cur_band_type);

            int group;

            if (cur_band_type == ZERO_BT) {

                for (group = 0; group < ics->group_len[g]; group++) {

                    memset(coef + group * 128 + offsets[i], 0, (offsets[i+1] - offsets[i])*sizeof(float));

                }

            }else if (cur_band_type == NOISE_BT) {

                const float scale = sf[idx] / ((offsets[i+1] - offsets[i]) * PNS_MEAN_ENERGY);

                for (group = 0; group < ics->group_len[g]; group++) {

                    for (k = offsets[i]; k < offsets[i+1]; k++) {

                        ac->random_state  = lcg_random(ac->random_state);

                        coef[group*128+k] = ac->random_state * scale;

                    }

                }

            }else if (cur_band_type != INTENSITY_BT2 && cur_band_type != INTENSITY_BT) {

                for (group = 0; group < ics->group_len[g]; group++) {

                    for (k = offsets[i]; k < offsets[i+1]; k += dim) {

                        const int index = get_vlc2(gb, vlc_spectral[cur_band_type - 1].table, 6, 3);

                        const int coef_tmp_idx = (group << 7) + k;

                        const float *vq_ptr;

                        int j;

                        if(index >= ff_aac_spectral_sizes[cur_band_type - 1]) {

                            av_log(ac->avccontext, AV_LOG_ERROR,

                                "Read beyond end of ff_aac_codebook_vectors[%d][]. index %d >= %d\n",

                                cur_band_type - 1, index, ff_aac_spectral_sizes[cur_band_type - 1]);

                            return -1;

                        }

                        vq_ptr = &ff_aac_codebook_vectors[cur_band_type - 1][index * dim];

                        if (is_cb_unsigned) {

                            for (j = 0; j < dim; j++)

                                if (vq_ptr[j])

                                    coef[coef_tmp_idx + j] = 1 - 2*(int)get_bits1(gb);

                        }else {

                            for (j = 0; j < dim; j++)

                                coef[coef_tmp_idx + j] = 1.0f;

                        }

                        if (cur_band_type == ESC_BT) {

                            for (j = 0; j < 2; j++) {

                                if (vq_ptr[j] == 64.0f) {

                                    int n = 4;

                                    /* The total length of escape_sequence must be < 22 bits according

                                       to the specification (i.e. max is 11111111110xxxxxxxxxx). */

                                    while (get_bits1(gb) && n < 15) n++;

                                    if(n == 15) {

                                        av_log(ac->avccontext, AV_LOG_ERROR, "error in spectral data, ESC overflow\n");

                                        return -1;

                                    }

                                    n = (1<<n) + get_bits(gb, n);

                                    coef[coef_tmp_idx + j] *= cbrtf(fabsf(n)) * n;

                                }else

                                    coef[coef_tmp_idx + j] *= vq_ptr[j];

                            }

                        }else

                            for (j = 0; j < dim; j++)

                                coef[coef_tmp_idx + j] *= vq_ptr[j];

                        for (j = 0; j < dim; j++)

                            coef[coef_tmp_idx + j] *= sf[idx];

                    }

                }

            }

        }

        coef += ics->group_len[g]<<7;

    }



    if (pulse_present) {

        for(i = 0; i < pulse->num_pulse; i++){

            float co  = coef_base[ pulse->pos[i] ];

            float ico = co / sqrtf(sqrtf(fabsf(co))) + pulse->amp[i];

            coef_base[ pulse->pos[i] ] = cbrtf(fabsf(ico)) * ico;

        }

    }

    return 0;

}
