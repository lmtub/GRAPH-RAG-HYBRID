static void chomp3(ChannelData *ctx, int16_t *output, uint8_t val,

                   const uint16_t tab1[],

                   const int16_t *tab2, int tab2_stride,

                   uint32_t numChannels)

{

    int16_t current;



    current = tab2[((ctx->index & 0x7f0) >> 4)*tab2_stride + val];



    current = mace_broken_clip_int16(current + ctx->lev);



    ctx->lev = current - (current >> 3);

    *output = QT_8S_2_16S(current);

    if (( ctx->index += tab1[val]-(ctx->index >> 5) ) < 0)

        ctx->index = 0;

}
