void ff_aac_search_for_tns(AACEncContext *s, SingleChannelElement *sce)

{

    TemporalNoiseShaping *tns = &sce->tns;

    int w, g, order, sfb_start, sfb_len, coef_start, shift[MAX_LPC_ORDER], count = 0;

    const int is8 = sce->ics.window_sequence[0] == EIGHT_SHORT_SEQUENCE;

    const int tns_max_order = is8 ? 7 : s->profile == FF_PROFILE_AAC_LOW ? 12 : TNS_MAX_ORDER;

    const float freq_mult = mpeg4audio_sample_rates[s->samplerate_index]/(1024.0f/sce->ics.num_windows)/2.0f;

    float max_coef = 0.0f;



    sce->tns.present = 0;

    return;



    for (coef_start = 0; coef_start < 1024; coef_start++)

        max_coef = FFMAX(max_coef, sce->pcoeffs[coef_start]);



    for (w = 0; w < sce->ics.num_windows; w++) {

        int filters = 1, start = 0, coef_len = 0;

        int32_t conv_coeff[1024] = {0};

        int32_t coefs_t[MAX_LPC_ORDER][MAX_LPC_ORDER] = {{0}};



        /* Determine start sfb + coef - excludes anything below threshold */

        for (g = 0;  g < sce->ics.num_swb; g++) {

            if (start*freq_mult > TNS_LOW_LIMIT) {

                sfb_start = w*16+g;

                sfb_len   = (w+1)*16 + g - sfb_start;

                coef_start = sce->ics.swb_offset[sfb_start];

                coef_len  = sce->ics.swb_offset[sfb_start + sfb_len] - coef_start;

                break;

            }

            start += sce->ics.swb_sizes[g];

        }



        if (coef_len <= 0)

            continue;



        conv_to_int32(conv_coeff, &sce->pcoeffs[coef_start], coef_len, max_coef);



        /* LPC */

        order = ff_lpc_calc_coefs(&s->lpc, conv_coeff, coef_len,

                                  TNS_MIN_PRED_ORDER, tns_max_order,

                                  32, coefs_t, shift,

                                  FF_LPC_TYPE_LEVINSON, 10,

                                  ORDER_METHOD_EST, MAX_LPC_SHIFT, 0) - 1;



        /* Works surprisingly well, remember to tweak MAX_LPC_SHIFT if you want to play around with this */

        if (shift[order] > 3) {

            int direction = 0;

            float tns_coefs_raw[TNS_MAX_ORDER];

            tns->n_filt[w] = filters++;

            conv_to_float(tns_coefs_raw, coefs_t[order], order);

            for (g = 0; g < tns->n_filt[w]; g++) {

                process_tns_coeffs(tns, tns_coefs_raw, order, w, g);

                apply_tns_filter(&sce->coeffs[coef_start], sce->pcoeffs, order, direction, tns->coef[w][g],

                                 sce->ics.ltp.present, w, g, coef_start, coef_len);

                tns->order[w][g]     = order;

                tns->length[w][g]    = sfb_len;

                tns->direction[w][g] = direction;

            }

            count++;

        }

    }



    sce->tns.present = !!count;

}
