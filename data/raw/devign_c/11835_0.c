static void entropy_decode(APEContext *ctx, int blockstodecode, int stereo)

{

    int32_t *decoded0 = ctx->decoded[0];

    int32_t *decoded1 = ctx->decoded[1];



    while (blockstodecode--) {

        *decoded0++ = ape_decode_value(ctx, &ctx->riceY);

        if (stereo)

            *decoded1++ = ape_decode_value(ctx, &ctx->riceX);

    }

}
