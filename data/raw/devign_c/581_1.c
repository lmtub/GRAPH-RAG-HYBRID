static void sbr_hf_assemble(int Y1[38][64][2],

                            const int X_high[64][40][2],

                            SpectralBandReplication *sbr, SBRData *ch_data,

                            const int e_a[2])

{

    int e, i, j, m;

    const int h_SL = 4 * !sbr->bs_smoothing_mode;

    const int kx = sbr->kx[1];

    const int m_max = sbr->m[1];

    static const SoftFloat h_smooth[5] = {

      { 715827883, -1 },

      { 647472402, -1 },

      { 937030863, -2 },

      { 989249804, -3 },

      { 546843842, -4 },

    };

    SoftFloat (*g_temp)[48] = ch_data->g_temp, (*q_temp)[48] = ch_data->q_temp;

    int indexnoise = ch_data->f_indexnoise;

    int indexsine  = ch_data->f_indexsine;



    if (sbr->reset) {

        for (i = 0; i < h_SL; i++) {

            memcpy(g_temp[i + 2*ch_data->t_env[0]], sbr->gain[0], m_max * sizeof(sbr->gain[0][0]));

            memcpy(q_temp[i + 2*ch_data->t_env[0]], sbr->q_m[0],  m_max * sizeof(sbr->q_m[0][0]));

        }

    } else if (h_SL) {

        for (i = 0; i < 4; i++) {

            memcpy(g_temp[i + 2 * ch_data->t_env[0]],

                   g_temp[i + 2 * ch_data->t_env_num_env_old],

                   sizeof(g_temp[0]));

            memcpy(q_temp[i + 2 * ch_data->t_env[0]],

                   q_temp[i + 2 * ch_data->t_env_num_env_old],

                   sizeof(q_temp[0]));

        }

    }



    for (e = 0; e < ch_data->bs_num_env; e++) {

        for (i = 2 * ch_data->t_env[e]; i < 2 * ch_data->t_env[e + 1]; i++) {

            memcpy(g_temp[h_SL + i], sbr->gain[e], m_max * sizeof(sbr->gain[0][0]));

            memcpy(q_temp[h_SL + i], sbr->q_m[e],  m_max * sizeof(sbr->q_m[0][0]));

        }

    }



    for (e = 0; e < ch_data->bs_num_env; e++) {

        for (i = 2 * ch_data->t_env[e]; i < 2 * ch_data->t_env[e + 1]; i++) {

            SoftFloat g_filt_tab[48];

            SoftFloat q_filt_tab[48];

            SoftFloat *g_filt, *q_filt;



            if (h_SL && e != e_a[0] && e != e_a[1]) {

                g_filt = g_filt_tab;

                q_filt = q_filt_tab;

                for (m = 0; m < m_max; m++) {

                    const int idx1 = i + h_SL;

                    g_filt[m].mant = g_filt[m].exp = 0;

                    q_filt[m].mant = q_filt[m].exp = 0;

                    for (j = 0; j <= h_SL; j++) {

                        g_filt[m] = av_add_sf(g_filt[m],

                                        av_mul_sf(g_temp[idx1 - j][m],

                                            h_smooth[j]));

                        q_filt[m] = av_add_sf(q_filt[m],

                                        av_mul_sf(q_temp[idx1 - j][m],

                                            h_smooth[j]));

                    }

                }

            } else {

                g_filt = g_temp[i + h_SL];

                q_filt = q_temp[i];

            }



            sbr->dsp.hf_g_filt(Y1[i] + kx, X_high + kx, g_filt, m_max,

                               i + ENVELOPE_ADJUSTMENT_OFFSET);



            if (e != e_a[0] && e != e_a[1]) {

                sbr->dsp.hf_apply_noise[indexsine](Y1[i] + kx, sbr->s_m[e],

                                                   q_filt, indexnoise,

                                                   kx, m_max);

            } else {

                int idx = indexsine&1;

                int A = (1-((indexsine+(kx & 1))&2));

                int B = (A^(-idx)) + idx;

                int *out = &Y1[i][kx][idx];

                int shift, round;



                SoftFloat *in  = sbr->s_m[e];

                for (m = 0; m+1 < m_max; m+=2) {

                    shift = 22 - in[m  ].exp;

                    if (shift < 32) {

                        round = 1 << (shift-1);

                        out[2*m  ] += (in[m  ].mant * A + round) >> shift;

                    }



                    shift = 22 - in[m+1].exp;

                    if (shift < 32) {

                        round = 1 << (shift-1);

                        out[2*m+2] += (in[m+1].mant * B + round) >> shift;

                    }

                }

                if(m_max&1)

                {

                    shift = 22 - in[m  ].exp;

                    if (shift < 32) {

                        round = 1 << (shift-1);

                        out[2*m  ] += (in[m  ].mant * A + round) >> shift;

                    }

                }

            }

            indexnoise = (indexnoise + m_max) & 0x1ff;

            indexsine = (indexsine + 1) & 3;

        }

    }

    ch_data->f_indexnoise = indexnoise;

    ch_data->f_indexsine  = indexsine;

}
