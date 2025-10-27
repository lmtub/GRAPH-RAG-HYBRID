static int sbr_hf_gen(AACContext *ac, SpectralBandReplication *sbr,

                      float X_high[64][40][2], const float X_low[32][40][2],

                      const float (*alpha0)[2], const float (*alpha1)[2],

                      const float bw_array[5], const uint8_t *t_env,

                      int bs_num_env)

{

    int i, j, x;

    int g = 0;

    int k = sbr->kx[1];

    for (j = 0; j < sbr->num_patches; j++) {

        for (x = 0; x < sbr->patch_num_subbands[j]; x++, k++) {

            float alpha[4];

            const int p = sbr->patch_start_subband[j] + x;

            while (g <= sbr->n_q && k >= sbr->f_tablenoise[g])

                g++;

            g--;



            if (g < 0) {

                av_log(ac->avctx, AV_LOG_ERROR,

                       "ERROR : no subband found for frequency %d\n", k);

                return -1;

            }



            alpha[0] = alpha1[p][0] * bw_array[g] * bw_array[g];

            alpha[1] = alpha1[p][1] * bw_array[g] * bw_array[g];

            alpha[2] = alpha0[p][0] * bw_array[g];

            alpha[3] = alpha0[p][1] * bw_array[g];



            for (i = 2 * t_env[0]; i < 2 * t_env[bs_num_env]; i++) {

                const int idx = i + ENVELOPE_ADJUSTMENT_OFFSET;

                X_high[k][idx][0] =

                    X_low[p][idx - 2][0] * alpha[0] -

                    X_low[p][idx - 2][1] * alpha[1] +

                    X_low[p][idx - 1][0] * alpha[2] -

                    X_low[p][idx - 1][1] * alpha[3] +

                    X_low[p][idx][0];

                X_high[k][idx][1] =

                    X_low[p][idx - 2][1] * alpha[0] +

                    X_low[p][idx - 2][0] * alpha[1] +

                    X_low[p][idx - 1][1] * alpha[2] +

                    X_low[p][idx - 1][0] * alpha[3] +

                    X_low[p][idx][1];

            }

        }

    }

    if (k < sbr->m[1] + sbr->kx[1])

        memset(X_high + k, 0, (sbr->m[1] + sbr->kx[1] - k) * sizeof(*X_high));



    return 0;

}
