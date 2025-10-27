static void ac3_downmix(AC3DecodeContext *s)

{

    int i, j;

    float v0, v1;



    for(i=0; i<256; i++) {

        v0 = v1 = 0.0f;

        for(j=0; j<s->fbw_channels; j++) {

            v0 += s->output[j][i] * s->downmix_coeffs[j][0];

            v1 += s->output[j][i] * s->downmix_coeffs[j][1];

        }

        v0 /= s->downmix_coeff_sum[0];

        v1 /= s->downmix_coeff_sum[1];

        if(s->output_mode == AC3_CHMODE_MONO) {

            s->output[0][i] = (v0 + v1) * LEVEL_MINUS_3DB;

        } else if(s->output_mode == AC3_CHMODE_STEREO) {

            s->output[0][i] = v0;

            s->output[1][i] = v1;

        }

    }

}
