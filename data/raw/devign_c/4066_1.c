void RENAME(swri_noise_shaping)(SwrContext *s, AudioData *dsts, const AudioData *srcs, const AudioData *noises, int count){

    int i, j, pos, ch;

    int taps  = s->dither.ns_taps;

    float S   = s->dither.ns_scale;

    float S_1 = s->dither.ns_scale_1;



    av_assert2((taps&3) != 2);

    av_assert2((taps&3) != 3 || s->dither.ns_coeffs[taps] == 0);



    for (ch=0; ch<srcs->ch_count; ch++) {

        const float *noise = ((const float *)noises->ch[ch]) + s->dither.noise_pos;

        const DELEM *src = (const DELEM*)srcs->ch[ch];

        DELEM *dst = (DELEM*)dsts->ch[ch];

        float *ns_errors = s->dither.ns_errors[ch];

        const float *ns_coeffs = s->dither.ns_coeffs;

        pos  = s->dither.ns_pos;

        for (i=0; i<count; i++) {

            double d1, d = src[i]*S_1;

            for(j=0; j<taps-2; j+=4) {

                d -= ns_coeffs[j    ] * ns_errors[pos + j    ]

                    +ns_coeffs[j + 1] * ns_errors[pos + j + 1]

                    +ns_coeffs[j + 2] * ns_errors[pos + j + 2]

                    +ns_coeffs[j + 3] * ns_errors[pos + j + 3];

            }

            if(j < taps)

                d -= ns_coeffs[j] * ns_errors[pos + j];

            pos = pos ? pos - 1 : taps - 1;

            d1 = rint(d + noise[i]);

            ns_errors[pos + taps] = ns_errors[pos] = d1 - d;

            d1 *= S;

            CLIP(d1);

            dst[i] = d1;

        }

    }



    s->dither.ns_pos = pos;

}
