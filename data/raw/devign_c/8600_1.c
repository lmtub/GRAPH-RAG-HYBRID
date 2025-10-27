void ff_ac3_bit_alloc_calc_mask(AC3BitAllocParameters *s, int16_t *band_psd,

                                int start, int end, int fast_gain, int is_lfe,

                                int dba_mode, int dba_nsegs, uint8_t *dba_offsets,

                                uint8_t *dba_lengths, uint8_t *dba_values,

                                int16_t *mask)

{

    int16_t excite[50]; /* excitation */

    int bin, k;

    int bndstrt, bndend, begin, end1, tmp;

    int lowcomp, fastleak, slowleak;



    /* excitation function */

    bndstrt = bin_to_band_tab[start];

    bndend = bin_to_band_tab[end-1] + 1;



    if (bndstrt == 0) {

        lowcomp = 0;

        lowcomp = calc_lowcomp1(lowcomp, band_psd[0], band_psd[1], 384);

        excite[0] = band_psd[0] - fast_gain - lowcomp;

        lowcomp = calc_lowcomp1(lowcomp, band_psd[1], band_psd[2], 384);

        excite[1] = band_psd[1] - fast_gain - lowcomp;

        begin = 7;

        for (bin = 2; bin < 7; bin++) {

            if (!(is_lfe && bin == 6))

                lowcomp = calc_lowcomp1(lowcomp, band_psd[bin], band_psd[bin+1], 384);

            fastleak = band_psd[bin] - fast_gain;

            slowleak = band_psd[bin] - s->slow_gain;

            excite[bin] = fastleak - lowcomp;

            if (!(is_lfe && bin == 6)) {

                if (band_psd[bin] <= band_psd[bin+1]) {

                    begin = bin + 1;

                    break;

                }

            }

        }



        end1=bndend;

        if (end1 > 22) end1=22;



        for (bin = begin; bin < end1; bin++) {

            if (!(is_lfe && bin == 6))

                lowcomp = calc_lowcomp(lowcomp, band_psd[bin], band_psd[bin+1], bin);



            fastleak = FFMAX(fastleak - s->fast_decay, band_psd[bin] - fast_gain);

            slowleak = FFMAX(slowleak - s->slow_decay, band_psd[bin] - s->slow_gain);

            excite[bin] = FFMAX(fastleak - lowcomp, slowleak);

        }

        begin = 22;

    } else {

        /* coupling channel */

        begin = bndstrt;



        fastleak = (s->cpl_fast_leak << 8) + 768;

        slowleak = (s->cpl_slow_leak << 8) + 768;

    }



    for (bin = begin; bin < bndend; bin++) {

        fastleak = FFMAX(fastleak - s->fast_decay, band_psd[bin] - fast_gain);

        slowleak = FFMAX(slowleak - s->slow_decay, band_psd[bin] - s->slow_gain);

        excite[bin] = FFMAX(fastleak, slowleak);

    }



    /* compute masking curve */



    for (bin = bndstrt; bin < bndend; bin++) {

        tmp = s->db_per_bit - band_psd[bin];

        if (tmp > 0) {

            excite[bin] += tmp >> 2;

        }

        mask[bin] = FFMAX(ff_ac3_hearing_threshold_tab[bin >> s->sr_shift][s->sr_code], excite[bin]);

    }



    /* delta bit allocation */



    if (dba_mode == DBA_REUSE || dba_mode == DBA_NEW) {

        int band, seg, delta;

        band = 0;

        for (seg = 0; seg < dba_nsegs; seg++) {

            band += dba_offsets[seg];

            if (dba_values[seg] >= 4) {

                delta = (dba_values[seg] - 3) << 7;

            } else {

                delta = (dba_values[seg] - 4) << 7;

            }

            for (k = 0; k < dba_lengths[seg]; k++) {

                mask[band] += delta;

                band++;

            }

        }

    }

}
