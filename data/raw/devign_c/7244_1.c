void ff_aac_search_for_pred(AACEncContext *s, SingleChannelElement *sce)

{

    int sfb, i, count = 0, cost_coeffs = 0, cost_pred = 0;

    const int pmax = FFMIN(sce->ics.max_sfb, ff_aac_pred_sfb_max[s->samplerate_index]);

    float *O34  = &s->scoefs[128*0], *P34 = &s->scoefs[128*1];

    float *SENT = &s->scoefs[128*2], *S34 = &s->scoefs[128*3];

    float *QERR = &s->scoefs[128*4];



    if (sce->ics.window_sequence[0] == EIGHT_SHORT_SEQUENCE) {

        sce->ics.predictor_present = 0;

        return;

    }



    if (!sce->ics.predictor_initialized) {

        reset_all_predictors(sce->predictor_state);

        sce->ics.predictor_initialized = 1;

        memcpy(sce->prcoeffs, sce->coeffs, 1024*sizeof(float));

        for (i = 1; i < 31; i++)

            sce->ics.predictor_reset_count[i] = i;

    }



    update_pred_resets(sce);

    memcpy(sce->band_alt, sce->band_type, sizeof(sce->band_type));



    for (sfb = PRED_SFB_START; sfb < pmax; sfb++) {

        int cost1, cost2, cb_p;

        float dist1, dist2, dist_spec_err = 0.0f;

        const int cb_n = sce->band_type[sfb];

        const int start_coef = sce->ics.swb_offset[sfb];

        const int num_coeffs = sce->ics.swb_offset[sfb + 1] - start_coef;

        const FFPsyBand *band = &s->psy.ch[s->cur_channel].psy_bands[sfb];



        if (start_coef + num_coeffs > MAX_PREDICTORS ||

            (s->cur_channel && sce->band_type[sfb] >= INTENSITY_BT2) ||

            sce->band_type[sfb] == NOISE_BT)

            continue;



        /* Normal coefficients */

        abs_pow34_v(O34, &sce->coeffs[start_coef], num_coeffs);

        dist1 = quantize_and_encode_band_cost(s, NULL, &sce->coeffs[start_coef], NULL,

                                              O34, num_coeffs, sce->sf_idx[sfb],

                                              cb_n, s->lambda / band->threshold, INFINITY, &cost1, 0);

        cost_coeffs += cost1;



        /* Encoded coefficients - needed for #bits, band type and quant. error */

        for (i = 0; i < num_coeffs; i++)

            SENT[i] = sce->coeffs[start_coef + i] - sce->prcoeffs[start_coef + i];

        abs_pow34_v(S34, SENT, num_coeffs);

        if (cb_n < RESERVED_BT)

            cb_p = find_min_book(find_max_val(1, num_coeffs, S34), sce->sf_idx[sfb]);

        else

            cb_p = cb_n;

        quantize_and_encode_band_cost(s, NULL, SENT, QERR, S34, num_coeffs,

                                      sce->sf_idx[sfb], cb_p, s->lambda / band->threshold, INFINITY,

                                      &cost2, 0);



        /* Reconstructed coefficients - needed for distortion measurements */

        for (i = 0; i < num_coeffs; i++)

            sce->prcoeffs[start_coef + i] += QERR[i] != 0.0f ? (sce->prcoeffs[start_coef + i] - QERR[i]) : 0.0f;

        abs_pow34_v(P34, &sce->prcoeffs[start_coef], num_coeffs);

        if (cb_n < RESERVED_BT)

            cb_p = find_min_book(find_max_val(1, num_coeffs, P34), sce->sf_idx[sfb]);

        else

            cb_p = cb_n;

        dist2 = quantize_and_encode_band_cost(s, NULL, &sce->prcoeffs[start_coef], NULL,

                                              P34, num_coeffs, sce->sf_idx[sfb],

                                              cb_p, s->lambda / band->threshold, INFINITY, NULL, 0);

        for (i = 0; i < num_coeffs; i++)

            dist_spec_err += (O34[i] - P34[i])*(O34[i] - P34[i]);

        dist_spec_err *= s->lambda / band->threshold;

        dist2 += dist_spec_err;



        if (dist2 <= dist1 && cb_p <= cb_n) {

            cost_pred += cost2;

            sce->ics.prediction_used[sfb] = 1;

            sce->band_alt[sfb]  = cb_n;

            sce->band_type[sfb] = cb_p;

            count++;

        } else {

            cost_pred += cost1;

            sce->band_alt[sfb] = cb_p;

        }

    }



    if (count && cost_coeffs < cost_pred) {

        count = 0;

        for (sfb = PRED_SFB_START; sfb < pmax; sfb++)

            RESTORE_PRED(sce, sfb);

        memset(&sce->ics.prediction_used, 0, sizeof(sce->ics.prediction_used));

    }



    sce->ics.predictor_present = !!count;

}
