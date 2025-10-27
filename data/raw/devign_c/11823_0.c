static inline void mix_3f_2r_to_mono(AC3DecodeContext *ctx)

{

    int i;

    float (*output)[256] = ctx->audio_block.block_output;



    for (i = 0; i < 256; i++)

        output[1][i] += (output[2][i] + output[3][i] + output[4][i] + output[5][i]);

    memset(output[2], 0, sizeof(output[2]));

    memset(output[3], 0, sizeof(output[3]));

    memset(output[4], 0, sizeof(output[4]));

    memset(output[5], 0, sizeof(output[5]));

}
