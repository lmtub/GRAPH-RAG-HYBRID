av_cold void ff_psy_preprocess_end(struct FFPsyPreprocessContext *ctx)

{

    int i;

    ff_iir_filter_free_coeffs(ctx->fcoeffs);

    if (ctx->fstate)

        for (i = 0; i < ctx->avctx->channels; i++)

            ff_iir_filter_free_state(ctx->fstate[i]);

    av_freep(&ctx->fstate);


}