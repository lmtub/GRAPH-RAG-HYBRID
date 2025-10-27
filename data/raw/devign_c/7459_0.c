static int synth_superframe(AVCodecContext *ctx,

                            float *samples, int *data_size)

{

    WMAVoiceContext *s = ctx->priv_data;

    GetBitContext *gb = &s->gb, s_gb;

    int n, res, n_samples = 480;

    double lsps[MAX_FRAMES][MAX_LSPS];

    const double *mean_lsf = s->lsps == 16 ?

        wmavoice_mean_lsf16[s->lsp_def_mode] : wmavoice_mean_lsf10[s->lsp_def_mode];

    float excitation[MAX_SIGNAL_HISTORY + MAX_SFRAMESIZE + 12];

    float synth[MAX_LSPS + MAX_SFRAMESIZE];



    memcpy(synth,      s->synth_history,

           s->lsps             * sizeof(*synth));

    memcpy(excitation, s->excitation_history,

           s->history_nsamples * sizeof(*excitation));



    if (s->sframe_cache_size > 0) {

        gb = &s_gb;

        init_get_bits(gb, s->sframe_cache, s->sframe_cache_size);

        s->sframe_cache_size = 0;

    }



    if ((res = check_bits_for_superframe(gb, s)) == 1) {

        *data_size = 0;

        return 1;

    }



    /* First bit is speech/music bit, it differentiates between WMAVoice

     * speech samples (the actual codec) and WMAVoice music samples, which

     * are really WMAPro-in-WMAVoice-superframes. I've never seen those in

     * the wild yet. */

    if (!get_bits1(gb)) {

        av_log_missing_feature(ctx, "WMAPro-in-WMAVoice support", 1);

        return -1;

    }



    /* (optional) nr. of samples in superframe; always <= 480 and >= 0 */

    if (get_bits1(gb)) {

        if ((n_samples = get_bits(gb, 12)) > 480) {

            av_log(ctx, AV_LOG_ERROR,

                   "Superframe encodes >480 samples (%d), not allowed\n",

                   n_samples);

            return -1;

        }

    }

    /* Parse LSPs, if global for the superframe (can also be per-frame). */

    if (s->has_residual_lsps) {

        double prev_lsps[MAX_LSPS], a1[MAX_LSPS * 2], a2[MAX_LSPS * 2];



        for (n = 0; n < s->lsps; n++)

            prev_lsps[n] = s->prev_lsps[n] - mean_lsf[n];



        if (s->lsps == 10) {

            dequant_lsp10r(gb, lsps[2], prev_lsps, a1, a2, s->lsp_q_mode);

        } else /* s->lsps == 16 */

            dequant_lsp16r(gb, lsps[2], prev_lsps, a1, a2, s->lsp_q_mode);



        for (n = 0; n < s->lsps; n++) {

            lsps[0][n]  = mean_lsf[n] + (a1[n]           - a2[n * 2]);

            lsps[1][n]  = mean_lsf[n] + (a1[s->lsps + n] - a2[n * 2 + 1]);

            lsps[2][n] += mean_lsf[n];

        }

        for (n = 0; n < 3; n++)

            stabilize_lsps(lsps[n], s->lsps);

    }



    /* Parse frames, optionally preceeded by per-frame (independent) LSPs. */

    for (n = 0; n < 3; n++) {

        if (!s->has_residual_lsps) {

            int m;



            if (s->lsps == 10) {

                dequant_lsp10i(gb, lsps[n]);

            } else /* s->lsps == 16 */

                dequant_lsp16i(gb, lsps[n]);



            for (m = 0; m < s->lsps; m++)

                lsps[n][m] += mean_lsf[m];

            stabilize_lsps(lsps[n], s->lsps);

        }



        if ((res = synth_frame(ctx, gb, n,

                               &samples[n * MAX_FRAMESIZE],

                               lsps[n], n == 0 ? s->prev_lsps : lsps[n - 1],

                               &excitation[s->history_nsamples + n * MAX_FRAMESIZE],

                               &synth[s->lsps + n * MAX_FRAMESIZE]))) {

            *data_size = 0;

            return res;

        }

    }



    /* Statistics? FIXME - we don't check for length, a slight overrun

     * will be caught by internal buffer padding, and anything else

     * will be skipped, not read. */

    if (get_bits1(gb)) {

        res = get_bits(gb, 4);

        skip_bits(gb, 10 * (res + 1));

    }



    /* Specify nr. of output samples */

    *data_size = n_samples * sizeof(float);



    /* Update history */

    memcpy(s->prev_lsps,           lsps[2],

           s->lsps             * sizeof(*s->prev_lsps));

    memcpy(s->synth_history,      &synth[MAX_SFRAMESIZE],

           s->lsps             * sizeof(*synth));

    memcpy(s->excitation_history, &excitation[MAX_SFRAMESIZE],

           s->history_nsamples * sizeof(*excitation));

    if (s->do_apf)

        memmove(s->zero_exc_pf,       &s->zero_exc_pf[MAX_SFRAMESIZE],

                s->history_nsamples * sizeof(*s->zero_exc_pf));



    return 0;

}
