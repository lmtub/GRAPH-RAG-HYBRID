static inline void mix_dualmono_to_stereo(AC3DecodeContext *ctx)

{

    int i;

    float tmp;

    float (*output)[256] = ctx->audio_block.block_output;



    for (i = 0; i < 256; i++) {

        tmp = output[1][i] + output[2][i];

        output[1][i] = output[2][i] = tmp;

    }

}
