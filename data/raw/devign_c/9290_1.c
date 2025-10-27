static void chomp6(ChannelData *ctx, int16_t *output, uint8_t val,

                   const uint16_t tab1[],

                   const int16_t *tab2, int tab2_stride,

                   uint32_t numChannels)

{

    int16_t current;



    current = tab2[((ctx->index & 0x7f0) >> 4)*tab2_stride + val];



    if ((ctx->previous ^ current) >= 0) {

        ctx->factor = FFMIN(ctx->factor + 506, 32767);

    } else {

        if (ctx->factor - 314 < -32768)

            ctx->factor = -32767;

        else

            ctx->factor -= 314;

    }



    current = mace_broken_clip_int16(current + ctx->level);



    ctx->level = ((current*ctx->factor) >> 15);

    current >>= 1;



    output[0] = QT_8S_2_16S(ctx->previous + ctx->prev2 -

                            ((ctx->prev2-current) >> 2));

    output[numChannels] = QT_8S_2_16S(ctx->previous + current +

                                      ((ctx->prev2-current) >> 2));

    ctx->prev2 = ctx->previous;

    ctx->previous = current;



    if ((ctx->index += tab1[val] - (ctx->index >> 5)) < 0)

        ctx->index = 0;

}
