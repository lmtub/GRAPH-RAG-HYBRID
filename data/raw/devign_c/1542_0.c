static inline void mix_dualmono_to_mono(AC3DecodeContext *ctx)

{

    int i;

    float (*output)[256] = ctx->audio_block.block_output;



    for (i = 0; i < 256; i++)

        output[1][i] += output[2][i];

    memset(output[2], 0, sizeof(output[2]));

}
