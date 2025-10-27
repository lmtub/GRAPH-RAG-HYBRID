static inline void upmix_mono_to_stereo(AC3DecodeContext *ctx)

{

    int i;

    float (*output)[256] = ctx->audio_block.block_output;



    for (i = 0; i < 256; i++)

        output[2][i] = output[1][i];

}
