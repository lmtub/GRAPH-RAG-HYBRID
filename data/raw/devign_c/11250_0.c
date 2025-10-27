static void ape_unpack_mono(APEContext *ctx, int count)

{

    if (ctx->frameflags & APE_FRAMECODE_STEREO_SILENCE) {

        /* We are pure silence, so we're done. */

        av_log(ctx->avctx, AV_LOG_DEBUG, "pure silence mono\n");

        return;

    }



    entropy_decode(ctx, count, 0);

    ape_apply_filters(ctx, ctx->decoded[0], NULL, count);



    /* Now apply the predictor decoding */

    predictor_decode_mono(ctx, count);



    /* Pseudo-stereo - just copy left channel to right channel */

    if (ctx->channels == 2) {

        memcpy(ctx->decoded[1], ctx->decoded[0], count * sizeof(*ctx->decoded[1]));

    }

}
