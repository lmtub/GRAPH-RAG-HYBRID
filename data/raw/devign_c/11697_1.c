static void search_for_pns(AACEncContext *s, AVCodecContext *avctx, SingleChannelElement *sce)

{

    FFPsyBand *band;

    int w, g, w2, i;

    float *PNS = &s->scoefs[0*128], *PNS34 = &s->scoefs[1*128];

    float *NOR34 = &s->scoefs[3*128];

    const float lambda = s->lambda;

    const float freq_mult = avctx->sample_rate/(1024.0f/sce->ics.num_windows)/2.0f;

    const float thr_mult = NOISE_LAMBDA_REPLACE*(100.0f/lambda);

    const float spread_threshold = NOISE_SPREAD_THRESHOLD*(lambda/100.f);



    if (sce->ics.window_sequence[0] == EIGHT_SHORT_SEQUENCE)

        return;



    for (w = 0; w < sce->ics.num_windows; w += sce->ics.group_len[w]) {

        for (g = 0;  g < sce->ics.num_swb; g++) {

            int noise_sfi;

            float dist1 = 0.0f, dist2 = 0.0f, noise_amp;

            float pns_energy = 0.0f, energy_ratio, dist_thresh;

            float sfb_energy = 0.0f, threshold = 0.0f, spread = 0.0f;

            const int start = sce->ics.swb_offset[w*16+g];

            const float freq = start*freq_mult;

            const float freq_boost = FFMAX(0.88f*freq/NOISE_LOW_LIMIT, 1.0f);

            if (freq < NOISE_LOW_LIMIT || avctx->cutoff && freq >= avctx->cutoff)

                continue;

            for (w2 = 0; w2 < sce->ics.group_len[w]; w2++) {

                band = &s->psy.ch[s->cur_channel].psy_bands[(w+w2)*16+g];

                sfb_energy += band->energy;

                spread     += band->spread;

                threshold  += band->threshold;

            }



            /* Ramps down at ~8000Hz and loosens the dist threshold */

            dist_thresh = FFMIN(2.5f*NOISE_LOW_LIMIT/freq, 1.27f);



            if (sce->zeroes[w*16+g] || spread < spread_threshold ||

                sfb_energy > threshold*thr_mult*freq_boost) {

                sce->pns_ener[w*16+g] = sfb_energy;

                continue;

            }



            noise_sfi = av_clip(roundf(log2f(sfb_energy)*2), -100, 155); /* Quantize */

            noise_amp = -ff_aac_pow2sf_tab[noise_sfi + POW_SF2_ZERO];    /* Dequantize */

            for (w2 = 0; w2 < sce->ics.group_len[w]; w2++) {

                float band_energy, scale;

                const int start_c = sce->ics.swb_offset[(w+w2)*16+g];

                band = &s->psy.ch[s->cur_channel].psy_bands[(w+w2)*16+g];

                for (i = 0; i < sce->ics.swb_sizes[g]; i++)

                    PNS[i] = s->random_state = lcg_random(s->random_state);

                band_energy = s->fdsp->scalarproduct_float(PNS, PNS, sce->ics.swb_sizes[g]);

                scale = noise_amp/sqrtf(band_energy);

                s->fdsp->vector_fmul_scalar(PNS, PNS, scale, sce->ics.swb_sizes[g]);

                pns_energy += s->fdsp->scalarproduct_float(PNS, PNS, sce->ics.swb_sizes[g]);

                abs_pow34_v(NOR34, &sce->coeffs[start_c], sce->ics.swb_sizes[g]);

                abs_pow34_v(PNS34, PNS, sce->ics.swb_sizes[g]);

                dist1 += quantize_band_cost(s, &sce->coeffs[start_c],

                                            NOR34,

                                            sce->ics.swb_sizes[g],

                                            sce->sf_idx[(w+w2)*16+g],

                                            sce->band_alt[(w+w2)*16+g],

                                            lambda/band->threshold, INFINITY, NULL, 0);

                dist2 += quantize_band_cost(s, PNS,

                                            PNS34,

                                            sce->ics.swb_sizes[g],

                                            noise_sfi,

                                            NOISE_BT,

                                            lambda/band->threshold, INFINITY, NULL, 0);

            }

            energy_ratio = sfb_energy/pns_energy; /* Compensates for quantization error */

            sce->pns_ener[w*16+g] = energy_ratio*sfb_energy;

            if (energy_ratio > 0.85f && energy_ratio < 1.25f && dist1/dist2 > dist_thresh) {

                sce->band_type[w*16+g] = NOISE_BT;

                sce->zeroes[w*16+g] = 0;

                if (sce->band_type[w*16+g-1] != NOISE_BT && /* Prevent holes */

                    sce->band_type[w*16+g-2] == NOISE_BT) {

                    sce->band_type[w*16+g-1] = NOISE_BT;

                    sce->zeroes[w*16+g-1] = 0;

                }

            }

        }

    }

}
