static void sbr_gain_calc(AACContext *ac, SpectralBandReplication *sbr,

                          SBRData *ch_data, const int e_a[2])

{

    int e, k, m;

    // max gain limits : -3dB, 0dB, 3dB, inf dB (limiter off)

    static const SoftFloat limgain[4] = { { 760155524,  0 }, { 0x20000000,  1 },

                                            { 758351638,  1 }, { 625000000, 34 } };



    for (e = 0; e < ch_data->bs_num_env; e++) {

        int delta = !((e == e_a[1]) || (e == e_a[0]));

        for (k = 0; k < sbr->n_lim; k++) {

            SoftFloat gain_boost, gain_max;

            SoftFloat sum[2];

            sum[0] = sum[1] = FLOAT_0;

            for (m = sbr->f_tablelim[k] - sbr->kx[1]; m < sbr->f_tablelim[k + 1] - sbr->kx[1]; m++) {

                const SoftFloat temp = av_div_sf(sbr->e_origmapped[e][m],

                                            av_add_sf(FLOAT_1, sbr->q_mapped[e][m]));

                sbr->q_m[e][m] = av_sqrt_sf(av_mul_sf(temp, sbr->q_mapped[e][m]));

                sbr->s_m[e][m] = av_sqrt_sf(av_mul_sf(temp, av_int2sf(ch_data->s_indexmapped[e + 1][m], 0)));

                if (!sbr->s_mapped[e][m]) {

                    if (delta) {

                      sbr->gain[e][m] = av_sqrt_sf(av_div_sf(sbr->e_origmapped[e][m],

                                            av_mul_sf(av_add_sf(FLOAT_1, sbr->e_curr[e][m]),

                                            av_add_sf(FLOAT_1, sbr->q_mapped[e][m]))));

                    } else {

                      sbr->gain[e][m] = av_sqrt_sf(av_div_sf(sbr->e_origmapped[e][m],

                                            av_add_sf(FLOAT_1, sbr->e_curr[e][m])));

                    }

                } else {

                    sbr->gain[e][m] = av_sqrt_sf(

                                        av_div_sf(

                                            av_mul_sf(sbr->e_origmapped[e][m], sbr->q_mapped[e][m]),

                                            av_mul_sf(

                                                av_add_sf(FLOAT_1, sbr->e_curr[e][m]),

                                                av_add_sf(FLOAT_1, sbr->q_mapped[e][m]))));

                }


            }

            for (m = sbr->f_tablelim[k] - sbr->kx[1]; m < sbr->f_tablelim[k + 1] - sbr->kx[1]; m++) {

                sum[0] = av_add_sf(sum[0], sbr->e_origmapped[e][m]);

                sum[1] = av_add_sf(sum[1], sbr->e_curr[e][m]);

            }

            gain_max = av_mul_sf(limgain[sbr->bs_limiter_gains],

                            av_sqrt_sf(

                                av_div_sf(

                                    av_add_sf(FLOAT_EPSILON, sum[0]),

                                    av_add_sf(FLOAT_EPSILON, sum[1]))));

            if (av_gt_sf(gain_max, FLOAT_100000))

              gain_max = FLOAT_100000;

            for (m = sbr->f_tablelim[k] - sbr->kx[1]; m < sbr->f_tablelim[k + 1] - sbr->kx[1]; m++) {

                SoftFloat q_m_max = av_div_sf(

                                        av_mul_sf(sbr->q_m[e][m], gain_max),

                                        sbr->gain[e][m]);

                if (av_gt_sf(sbr->q_m[e][m], q_m_max))

                  sbr->q_m[e][m] = q_m_max;

                if (av_gt_sf(sbr->gain[e][m], gain_max))

                  sbr->gain[e][m] = gain_max;

            }

            sum[0] = sum[1] = FLOAT_0;

            for (m = sbr->f_tablelim[k] - sbr->kx[1]; m < sbr->f_tablelim[k + 1] - sbr->kx[1]; m++) {

                sum[0] = av_add_sf(sum[0], sbr->e_origmapped[e][m]);

                sum[1] = av_add_sf(sum[1],

                            av_mul_sf(

                                av_mul_sf(sbr->e_curr[e][m],

                                          sbr->gain[e][m]),

                                sbr->gain[e][m]));

                sum[1] = av_add_sf(sum[1],

                            av_mul_sf(sbr->s_m[e][m], sbr->s_m[e][m]));

                if (delta && !sbr->s_m[e][m].mant)

                  sum[1] = av_add_sf(sum[1],

                                av_mul_sf(sbr->q_m[e][m], sbr->q_m[e][m]));

            }

            gain_boost = av_sqrt_sf(

                            av_div_sf(

                                av_add_sf(FLOAT_EPSILON, sum[0]),

                                av_add_sf(FLOAT_EPSILON, sum[1])));

            if (av_gt_sf(gain_boost, FLOAT_1584893192))

              gain_boost = FLOAT_1584893192;



            for (m = sbr->f_tablelim[k] - sbr->kx[1]; m < sbr->f_tablelim[k + 1] - sbr->kx[1]; m++) {

                sbr->gain[e][m] = av_mul_sf(sbr->gain[e][m], gain_boost);

                sbr->q_m[e][m]  = av_mul_sf(sbr->q_m[e][m], gain_boost);

                sbr->s_m[e][m]  = av_mul_sf(sbr->s_m[e][m], gain_boost);

            }

        }

    }

}