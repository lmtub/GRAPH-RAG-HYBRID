static void search_for_pns(AACEncContext *s, AVCodecContext *avctx, SingleChannelElement *sce)

{

    int start = 0, w, w2, g;

    const float lambda = s->lambda;

    const float freq_mult = avctx->sample_rate/(1024.0f/sce->ics.num_windows)/2.0f;

    const float spread_threshold = NOISE_SPREAD_THRESHOLD*(lambda/120.f);

    const float thr_mult = NOISE_LAMBDA_NUMERATOR/lambda;



    for (w = 0; w < sce->ics.num_windows; w += sce->ics.group_len[w]) {

        start = 0;

        for (g = 0;  g < sce->ics.num_swb; g++) {

            if (start*freq_mult > NOISE_LOW_LIMIT*(lambda/170.0f)) {

                float energy = 0.0f, threshold = 0.0f, spread = 0.0f;

                for (w2 = 0; w2 < sce->ics.group_len[w]; w2++) {

                    FFPsyBand *band = &s->psy.ch[s->cur_channel+0].psy_bands[(w+w2)*16+g];

                    energy += band->energy;

                    threshold += band->threshold;

                    spread += band->spread;

                }

                if (spread > spread_threshold*sce->ics.group_len[w] &&

                    ((sce->zeroes[w*16+g] && energy >= threshold) ||

                    energy < threshold*thr_mult*sce->ics.group_len[w])) {

                    sce->band_type[w*16+g] = NOISE_BT;

                    sce->pns_ener[w*16+g] = energy / sce->ics.group_len[w];

                    sce->zeroes[w*16+g] = 0;

                }

            }

            start += sce->ics.swb_sizes[g];

        }

    }

}
