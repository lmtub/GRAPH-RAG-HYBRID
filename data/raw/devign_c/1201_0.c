static void psy_3gpp_analyze(FFPsyContext *ctx, int channel,

                             const float *coefs, const FFPsyWindowInfo *wi)

{

    AacPsyContext *pctx = (AacPsyContext*) ctx->model_priv_data;

    AacPsyChannel *pch  = &pctx->ch[channel];

    int start = 0;

    int i, w, g;

    const int num_bands       = ctx->num_bands[wi->num_windows == 8];

    const uint8_t* band_sizes = ctx->bands[wi->num_windows == 8];

    AacPsyCoeffs *coeffs     = &pctx->psy_coef[wi->num_windows == 8];



    //calculate energies, initial thresholds and related values - 5.4.2 "Threshold Calculation"

    for (w = 0; w < wi->num_windows*16; w += 16) {

        for (g = 0; g < num_bands; g++) {

            AacPsyBand *band = &pch->band[w+g];

            band->energy = 0.0f;

            for (i = 0; i < band_sizes[g]; i++)

                band->energy += coefs[start+i] * coefs[start+i];

            band->thr     = band->energy * 0.001258925f;

            start        += band_sizes[g];



            ctx->psy_bands[channel*PSY_MAX_BANDS+w+g].energy = band->energy;

        }

    }

    //modify thresholds - spread, threshold in quiet - 5.4.3 "Spreaded Energy Calculation"

    for (w = 0; w < wi->num_windows*16; w += 16) {

        AacPsyBand *band = &pch->band[w];

        for (g = 1; g < num_bands; g++)

            band[g].thr = FFMAX(band[g].thr, band[g-1].thr * coeffs->spread_hi [g]);

        for (g = num_bands - 2; g >= 0; g--)

            band[g].thr = FFMAX(band[g].thr, band[g+1].thr * coeffs->spread_low[g]);

        for (g = 0; g < num_bands; g++) {

            band[g].thr_quiet = band[g].thr = FFMAX(band[g].thr, coeffs->ath[g]);

            if (!(wi->window_type[0] == LONG_STOP_SEQUENCE || (wi->window_type[1] == LONG_START_SEQUENCE && !w)))

                band[g].thr = FFMAX(PSY_3GPP_RPEMIN*band[g].thr, FFMIN(band[g].thr,

                                    PSY_3GPP_RPELEV*pch->prev_band[w+g].thr_quiet));



            ctx->psy_bands[channel*PSY_MAX_BANDS+w+g].threshold = band[g].thr;

        }

    }

    memcpy(pch->prev_band, pch->band, sizeof(pch->band));

}
