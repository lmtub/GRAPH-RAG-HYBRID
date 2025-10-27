void ff_sbr_apply(AACContext *ac, SpectralBandReplication *sbr, int id_aac,

                  float* L, float* R)

{

    int downsampled = ac->m4ac.ext_sample_rate < sbr->sample_rate;

    int ch;

    int nch = (id_aac == TYPE_CPE) ? 2 : 1;



    if (sbr->start) {

        sbr_dequant(sbr, id_aac);

    }

    for (ch = 0; ch < nch; ch++) {

        /* decode channel */

        sbr_qmf_analysis(&ac->dsp, &sbr->mdct_ana, ch ? R : L, sbr->data[ch].analysis_filterbank_samples,

                         (float*)sbr->qmf_filter_scratch,

                         sbr->data[ch].W);

        sbr_lf_gen(ac, sbr, sbr->X_low, sbr->data[ch].W);

        if (sbr->start) {

            sbr_hf_inverse_filter(sbr->alpha0, sbr->alpha1, sbr->X_low, sbr->k[0]);

            sbr_chirp(sbr, &sbr->data[ch]);

            sbr_hf_gen(ac, sbr, sbr->X_high, sbr->X_low, sbr->alpha0, sbr->alpha1,

                       sbr->data[ch].bw_array, sbr->data[ch].t_env,

                       sbr->data[ch].bs_num_env);



            // hf_adj

            sbr_mapping(ac, sbr, &sbr->data[ch], sbr->data[ch].e_a);

            sbr_env_estimate(sbr->e_curr, sbr->X_high, sbr, &sbr->data[ch]);

            sbr_gain_calc(ac, sbr, &sbr->data[ch], sbr->data[ch].e_a);

            sbr_hf_assemble(sbr->data[ch].Y, sbr->X_high, sbr, &sbr->data[ch],

                            sbr->data[ch].e_a);

        }



        /* synthesis */

        sbr_x_gen(sbr, sbr->X[ch], sbr->X_low, sbr->data[ch].Y, ch);

    }



    if (ac->m4ac.ps == 1) {

        if (sbr->ps.start) {

            ff_ps_apply(ac->avctx, &sbr->ps, sbr->X[0], sbr->X[1], sbr->kx[1] + sbr->m[1]);

        } else {

            memcpy(sbr->X[1], sbr->X[0], sizeof(sbr->X[0]));

        }

        nch = 2;

    }



    sbr_qmf_synthesis(&ac->dsp, &sbr->mdct, L, sbr->X[0], sbr->qmf_filter_scratch,

                      sbr->data[0].synthesis_filterbank_samples,

                      &sbr->data[0].synthesis_filterbank_samples_offset,

                      downsampled);

    if (nch == 2)

        sbr_qmf_synthesis(&ac->dsp, &sbr->mdct, R, sbr->X[1], sbr->qmf_filter_scratch,

                          sbr->data[1].synthesis_filterbank_samples,

                          &sbr->data[1].synthesis_filterbank_samples_offset,

                          downsampled);

}
