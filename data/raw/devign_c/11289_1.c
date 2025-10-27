void ff_aac_search_for_is(AACEncContext *s, AVCodecContext *avctx, ChannelElement *cpe)

{

    SingleChannelElement *sce0 = &cpe->ch[0];

    SingleChannelElement *sce1 = &cpe->ch[1];

    int start = 0, count = 0, w, w2, g, i;

    const float freq_mult = avctx->sample_rate/(1024.0f/sce0->ics.num_windows)/2.0f;



    if (!cpe->common_window)

        return;



    for (w = 0; w < sce0->ics.num_windows; w += sce0->ics.group_len[w]) {

        start = 0;

        for (g = 0;  g < sce0->ics.num_swb; g++) {

            if (start*freq_mult > INT_STEREO_LOW_LIMIT*(s->lambda/170.0f) &&

                cpe->ch[0].band_type[w*16+g] != NOISE_BT && !cpe->ch[0].zeroes[w*16+g] &&

                cpe->ch[1].band_type[w*16+g] != NOISE_BT && !cpe->ch[1].zeroes[w*16+g]) {

                float ener0 = 0.0f, ener1 = 0.0f, ener01 = 0.0f;

                struct AACISError ph_err1, ph_err2, *erf;

                if (sce0->band_type[w*16+g] == NOISE_BT ||

                    sce1->band_type[w*16+g] == NOISE_BT) {

                    start += sce0->ics.swb_sizes[g];

                    continue;

                }

                for (w2 = 0; w2 < sce0->ics.group_len[w]; w2++) {

                    for (i = 0; i < sce0->ics.swb_sizes[g]; i++) {

                        float coef0 = fabsf(sce0->pcoeffs[start+(w+w2)*128+i]);

                        float coef1 = fabsf(sce1->pcoeffs[start+(w+w2)*128+i]);

                        ener0  += coef0*coef0;

                        ener1  += coef1*coef1;

                        ener01 += (coef0 + coef1)*(coef0 + coef1);

                    }

                }

                ph_err1 = ff_aac_is_encoding_err(s, cpe, start, w, g,

                                                 ener0, ener1, ener01, 0, -1);

                ph_err2 = ff_aac_is_encoding_err(s, cpe, start, w, g,

                                                 ener0, ener1, ener01, 0, +1);

                erf = ph_err1.error < ph_err2.error ? &ph_err1 : &ph_err2;

                if (erf->pass) {

                    cpe->is_mask[w*16+g] = 1;

                    cpe->ch[0].is_ener[w*16+g] = sqrt(ener0/ener01);

                    cpe->ch[1].is_ener[w*16+g] = ener0/ener1;

                    cpe->ch[1].band_type[w*16+g] = erf->phase ? INTENSITY_BT : INTENSITY_BT2;

                    count++;

                }

            }

            start += sce0->ics.swb_sizes[g];

        }

    }

    cpe->is_mode = !!count;

}
