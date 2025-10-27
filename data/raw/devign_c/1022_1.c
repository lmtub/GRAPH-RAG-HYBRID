static void psy_3gpp_analyze_channel(FFPsyContext *ctx, int channel,

                                     const float *coefs, const FFPsyWindowInfo *wi)

{

    AacPsyContext *pctx = (AacPsyContext*) ctx->model_priv_data;

    AacPsyChannel *pch  = &pctx->ch[channel];

    int start = 0;

    int i, w, g;

    float desired_bits, desired_pe, delta_pe, reduction, spread_en[128] = {0};

    float a = 0.0f, active_lines = 0.0f, norm_fac = 0.0f;

    float pe = pctx->chan_bitrate > 32000 ? 0.0f : FFMAX(50.0f, 100.0f - pctx->chan_bitrate * 100.0f / 32000.0f);

    const int      num_bands   = ctx->num_bands[wi->num_windows == 8];

    const uint8_t *band_sizes  = ctx->bands[wi->num_windows == 8];

    AacPsyCoeffs  *coeffs      = pctx->psy_coef[wi->num_windows == 8];

    const float avoid_hole_thr = wi->num_windows == 8 ? PSY_3GPP_AH_THR_SHORT : PSY_3GPP_AH_THR_LONG;



    //calculate energies, initial thresholds and related values - 5.4.2 "Threshold Calculation"

    for (w = 0; w < wi->num_windows*16; w += 16) {

        for (g = 0; g < num_bands; g++) {

            AacPsyBand *band = &pch->band[w+g];



            float form_factor = 0.0f;

            band->energy = 0.0f;

            for (i = 0; i < band_sizes[g]; i++) {

                band->energy += coefs[start+i] * coefs[start+i];

                form_factor  += sqrtf(fabs(coefs[start+i]));

            }

            band->thr      = band->energy * 0.001258925f;

            band->nz_lines = form_factor / powf(band->energy / band_sizes[g], 0.25f);



            start += band_sizes[g];

        }

    }

    //modify thresholds and energies - spread, threshold in quiet, pre-echo control

    for (w = 0; w < wi->num_windows*16; w += 16) {

        AacPsyBand *bands = &pch->band[w];



        //5.4.2.3 "Spreading" & 5.4.3 "Spreaded Energy Calculation"

        spread_en[0] = bands[0].energy;

        for (g = 1; g < num_bands; g++) {

            bands[g].thr   = FFMAX(bands[g].thr,    bands[g-1].thr * coeffs[g].spread_hi[0]);

            spread_en[w+g] = FFMAX(bands[g].energy, spread_en[w+g-1] * coeffs[g].spread_hi[1]);

        }

        for (g = num_bands - 2; g >= 0; g--) {

            bands[g].thr   = FFMAX(bands[g].thr,   bands[g+1].thr * coeffs[g].spread_low[0]);

            spread_en[w+g] = FFMAX(spread_en[w+g], spread_en[w+g+1] * coeffs[g].spread_low[1]);

        }

        //5.4.2.4 "Threshold in quiet"

        for (g = 0; g < num_bands; g++) {

            AacPsyBand *band = &bands[g];



            band->thr_quiet = band->thr = FFMAX(band->thr, coeffs[g].ath);

            //5.4.2.5 "Pre-echo control"

            if (!(wi->window_type[0] == LONG_STOP_SEQUENCE || (wi->window_type[1] == LONG_START_SEQUENCE && !w)))

                band->thr = FFMAX(PSY_3GPP_RPEMIN*band->thr, FFMIN(band->thr,

                                  PSY_3GPP_RPELEV*pch->prev_band[w+g].thr_quiet));



            /* 5.6.1.3.1 "Prepatory steps of the perceptual entropy calculation" */

            pe += calc_pe_3gpp(band);

            a  += band->pe_const;

            active_lines += band->active_lines;



            /* 5.6.1.3.3 "Selection of the bands for avoidance of holes" */

            if (spread_en[w+g] * avoid_hole_thr > band->energy || coeffs[g].min_snr > 1.0f)

                band->avoid_holes = PSY_3GPP_AH_NONE;

            else

                band->avoid_holes = PSY_3GPP_AH_INACTIVE;

        }

    }



    /* 5.6.1.3.2 "Calculation of the desired perceptual entropy" */

    ctx->ch[channel].entropy = pe;

    desired_bits = calc_bit_demand(pctx, pe, ctx->bitres.bits, ctx->bitres.size, wi->num_windows == 8);

    desired_pe = PSY_3GPP_BITS_TO_PE(desired_bits);

    /* NOTE: PE correction is kept simple. During initial testing it had very

     *       little effect on the final bitrate. Probably a good idea to come

     *       back and do more testing later.

     */

    if (ctx->bitres.bits > 0)

        desired_pe *= av_clipf(pctx->pe.previous / PSY_3GPP_BITS_TO_PE(ctx->bitres.bits),

                               0.85f, 1.15f);

    pctx->pe.previous = PSY_3GPP_BITS_TO_PE(desired_bits);



    if (desired_pe < pe) {

        /* 5.6.1.3.4 "First Estimation of the reduction value" */

        for (w = 0; w < wi->num_windows*16; w += 16) {

            reduction = calc_reduction_3gpp(a, desired_pe, pe, active_lines);

            pe = 0.0f;

            a  = 0.0f;

            active_lines = 0.0f;

            for (g = 0; g < num_bands; g++) {

                AacPsyBand *band = &pch->band[w+g];



                band->thr = calc_reduced_thr_3gpp(band, coeffs[g].min_snr, reduction);

                /* recalculate PE */

                pe += calc_pe_3gpp(band);

                a  += band->pe_const;

                active_lines += band->active_lines;

            }

        }



        /* 5.6.1.3.5 "Second Estimation of the reduction value" */

        for (i = 0; i < 2; i++) {

            float pe_no_ah = 0.0f, desired_pe_no_ah;

            active_lines = a = 0.0f;

            for (w = 0; w < wi->num_windows*16; w += 16) {

                for (g = 0; g < num_bands; g++) {

                    AacPsyBand *band = &pch->band[w+g];



                    if (band->avoid_holes != PSY_3GPP_AH_ACTIVE) {

                        pe_no_ah += band->pe;

                        a        += band->pe_const;

                        active_lines += band->active_lines;

                    }

                }

            }

            desired_pe_no_ah = FFMAX(desired_pe - (pe - pe_no_ah), 0.0f);

            if (active_lines > 0.0f)

                reduction += calc_reduction_3gpp(a, desired_pe_no_ah, pe_no_ah, active_lines);



            pe = 0.0f;

            for (w = 0; w < wi->num_windows*16; w += 16) {

                for (g = 0; g < num_bands; g++) {

                    AacPsyBand *band = &pch->band[w+g];



                    if (active_lines > 0.0f)

                        band->thr = calc_reduced_thr_3gpp(band, coeffs[g].min_snr, reduction);

                    pe += calc_pe_3gpp(band);

                    band->norm_fac = band->active_lines / band->thr;

                    norm_fac += band->norm_fac;

                }

            }

            delta_pe = desired_pe - pe;

            if (fabs(delta_pe) > 0.05f * desired_pe)

                break;

        }



        if (pe < 1.15f * desired_pe) {

            /* 6.6.1.3.6 "Final threshold modification by linearization" */

            norm_fac = 1.0f / norm_fac;

            for (w = 0; w < wi->num_windows*16; w += 16) {

                for (g = 0; g < num_bands; g++) {

                    AacPsyBand *band = &pch->band[w+g];



                    if (band->active_lines > 0.5f) {

                        float delta_sfb_pe = band->norm_fac * norm_fac * delta_pe;

                        float thr = band->thr;



                        thr *= powf(2.0f, delta_sfb_pe / band->active_lines);

                        if (thr > coeffs[g].min_snr * band->energy && band->avoid_holes == PSY_3GPP_AH_INACTIVE)

                            thr = FFMAX(band->thr, coeffs[g].min_snr * band->energy);

                        band->thr = thr;

                    }

                }

            }

        } else {

            /* 5.6.1.3.7 "Further perceptual entropy reduction" */

            g = num_bands;

            while (pe > desired_pe && g--) {

                for (w = 0; w < wi->num_windows*16; w+= 16) {

                    AacPsyBand *band = &pch->band[w+g];

                    if (band->avoid_holes != PSY_3GPP_AH_NONE && coeffs[g].min_snr < PSY_SNR_1DB) {

                        coeffs[g].min_snr = PSY_SNR_1DB;

                        band->thr = band->energy * PSY_SNR_1DB;

                        pe += band->active_lines * 1.5f - band->pe;

                    }

                }

            }

            /* TODO: allow more holes (unused without mid/side) */

        }

    }



    for (w = 0; w < wi->num_windows*16; w += 16) {

        for (g = 0; g < num_bands; g++) {

            AacPsyBand *band     = &pch->band[w+g];

            FFPsyBand  *psy_band = &ctx->ch[channel].psy_bands[w+g];



            psy_band->threshold = band->thr;

            psy_band->energy    = band->energy;

        }

    }



    memcpy(pch->prev_band, pch->band, sizeof(pch->band));

}
