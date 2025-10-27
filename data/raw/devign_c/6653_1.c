static void residual_interp(int16_t *buf, int16_t *out, int lag,

                            int gain, int *rseed)

{

    int i;

    if (lag) { /* Voiced */

        int16_t *vector_ptr = buf + PITCH_MAX;

        /* Attenuate */

        for (i = 0; i < lag; i++)

            out[i] = vector_ptr[i - lag] * 3 >> 2;

        av_memcpy_backptr((uint8_t*)(out + lag), lag * sizeof(*out),

                          (FRAME_LEN - lag) * sizeof(*out));

    } else {  /* Unvoiced */

        for (i = 0; i < FRAME_LEN; i++) {

            *rseed = *rseed * 521 + 259;

            out[i] = gain * *rseed >> 15;

        }

        memset(buf, 0, (FRAME_LEN + PITCH_MAX) * sizeof(*buf));

    }

}
