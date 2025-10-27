struct AACISError ff_aac_is_encoding_err(AACEncContext *s, ChannelElement *cpe,

                                         int start, int w, int g, float ener0,

                                         float ener1, float ener01,

                                         int use_pcoeffs, int phase)

{

    int i, w2;

    SingleChannelElement *sce0 = &cpe->ch[0];

    SingleChannelElement *sce1 = &cpe->ch[1];

    float *L = use_pcoeffs ? sce0->pcoeffs : sce0->coeffs;

    float *R = use_pcoeffs ? sce1->pcoeffs : sce1->coeffs;

    float *L34 = &s->scoefs[256*0], *R34 = &s->scoefs[256*1];

    float *IS  = &s->scoefs[256*2], *I34 = &s->scoefs[256*3];

    float dist1 = 0.0f, dist2 = 0.0f;

    struct AACISError is_error = {0};



    if (ener01 <= 0 || ener0 <= 0) {

        is_error.pass = 0;

        return is_error;

    }



    for (w2 = 0; w2 < sce0->ics.group_len[w]; w2++) {

        FFPsyBand *band0 = &s->psy.ch[s->cur_channel+0].psy_bands[(w+w2)*16+g];

        FFPsyBand *band1 = &s->psy.ch[s->cur_channel+1].psy_bands[(w+w2)*16+g];

        int is_band_type, is_sf_idx = FFMAX(1, sce0->sf_idx[(w+w2)*16+g]-4);

        float e01_34 = phase*pow(ener1/ener0, 3.0/4.0);

        float maxval, dist_spec_err = 0.0f;

        float minthr = FFMIN(band0->threshold, band1->threshold);

        for (i = 0; i < sce0->ics.swb_sizes[g]; i++)

            IS[i] = (L[start+(w+w2)*128+i] + phase*R[start+(w+w2)*128+i])*sqrt(ener0/ener01);

        abs_pow34_v(L34, &L[start+(w+w2)*128], sce0->ics.swb_sizes[g]);

        abs_pow34_v(R34, &R[start+(w+w2)*128], sce0->ics.swb_sizes[g]);

        abs_pow34_v(I34, IS,                   sce0->ics.swb_sizes[g]);

        maxval = find_max_val(1, sce0->ics.swb_sizes[g], I34);

        is_band_type = find_min_book(maxval, is_sf_idx);

        dist1 += quantize_band_cost(s, &L[start + (w+w2)*128], L34,

                                    sce0->ics.swb_sizes[g],

                                    sce0->sf_idx[(w+w2)*16+g],

                                    sce0->band_type[(w+w2)*16+g],

                                    s->lambda / band0->threshold, INFINITY, NULL, NULL, 0);

        dist1 += quantize_band_cost(s, &R[start + (w+w2)*128], R34,

                                    sce1->ics.swb_sizes[g],

                                    sce1->sf_idx[(w+w2)*16+g],

                                    sce1->band_type[(w+w2)*16+g],

                                    s->lambda / band1->threshold, INFINITY, NULL, NULL, 0);

        dist2 += quantize_band_cost(s, IS, I34, sce0->ics.swb_sizes[g],

                                    is_sf_idx, is_band_type,

                                    s->lambda / minthr, INFINITY, NULL, NULL, 0);

        for (i = 0; i < sce0->ics.swb_sizes[g]; i++) {

            dist_spec_err += (L34[i] - I34[i])*(L34[i] - I34[i]);

            dist_spec_err += (R34[i] - I34[i]*e01_34)*(R34[i] - I34[i]*e01_34);

        }

        dist_spec_err *= s->lambda / minthr;

        dist2 += dist_spec_err;

    }



    is_error.pass = dist2 <= dist1;

    is_error.phase = phase;

    is_error.error = fabsf(dist1 - dist2);

    is_error.dist1 = dist1;

    is_error.dist2 = dist2;

    is_error.ener01 = ener01;



    return is_error;

}
