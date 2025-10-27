void ff_aac_apply_tns(SingleChannelElement *sce)

{

    const int mmm = FFMIN(sce->ics.tns_max_bands, sce->ics.max_sfb);

    float *coef = sce->pcoeffs;

    TemporalNoiseShaping *tns = &sce->tns;

    int w, filt, m, i;

    int bottom, top, order, start, end, size, inc;

    float *lpc, tmp[TNS_MAX_ORDER+1];



    for (w = 0; w < sce->ics.num_windows; w++) {

        bottom = sce->ics.num_swb;

        for (filt = 0; filt < tns->n_filt[w]; filt++) {

            top    = bottom;

            bottom = FFMAX(0, top - tns->length[w][filt]);

            order  = tns->order[w][filt];

            lpc    = tns->coef[w][filt];

            if (!order)

                continue;



            start = sce->ics.swb_offset[FFMIN(bottom, mmm)];

            end   = sce->ics.swb_offset[FFMIN(   top, mmm)];

            if ((size = end - start) <= 0)

                continue;

            if (tns->direction[w][filt]) {

                inc = -1;

                start = end - 1;

            } else {

                inc = 1;

            }

            start += w * 128;



            if (!sce->ics.ltp.present) {

                // ar filter

                for (m = 0; m < size; m++, start += inc)

                    for (i = 1; i <= FFMIN(m, order); i++)

                        coef[start] += coef[start - i * inc]*lpc[i - 1];

            } else {

                // ma filter

                for (m = 0; m < size; m++, start += inc) {

                    tmp[0] = coef[start];

                    for (i = 1; i <= FFMIN(m, order); i++)

                        coef[start] += tmp[i]*lpc[i - 1];

                    for (i = order; i > 0; i--)

                        tmp[i] = tmp[i - 1];

                }

            }

        }

    }

}
