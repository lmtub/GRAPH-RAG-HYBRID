static int encode_rgb_frame(FFV1Context *s, uint8_t *src[3], int w, int h, int stride[3])

{

    int x, y, p, i;

    const int ring_size = s->avctx->context_model ? 3 : 2;

    int16_t *sample[4][3];

    int lbd    = s->bits_per_raw_sample <= 8;

    int bits   = s->bits_per_raw_sample > 0 ? s->bits_per_raw_sample : 8;

    int offset = 1 << bits;



    s->run_index = 0;



    memset(s->sample_buffer, 0, ring_size * MAX_PLANES *

                                (w + 6) * sizeof(*s->sample_buffer));



    for (y = 0; y < h; y++) {

        for (i = 0; i < ring_size; i++)

            for (p = 0; p < MAX_PLANES; p++)

                sample[p][i]= s->sample_buffer + p*ring_size*(w+6) + ((h+i-y)%ring_size)*(w+6) + 3;



        for (x = 0; x < w; x++) {

            int b, g, r, av_uninit(a);

            if (lbd) {

                unsigned v = *((uint32_t*)(src[0] + x*4 + stride[0]*y));

                b =  v        & 0xFF;

                g = (v >>  8) & 0xFF;

                r = (v >> 16) & 0xFF;

                a =  v >> 24;

            } else {

                b = *((uint16_t*)(src[0] + x*2 + stride[0]*y));

                g = *((uint16_t*)(src[1] + x*2 + stride[1]*y));

                r = *((uint16_t*)(src[2] + x*2 + stride[2]*y));

            }



            b -= g;

            r -= g;

            g += (b + r) >> 2;

            b += offset;

            r += offset;



            sample[0][0][x] = g;

            sample[1][0][x] = b;

            sample[2][0][x] = r;

            sample[3][0][x] = a;

        }

        for (p = 0; p < 3 + s->transparency; p++) {

            int ret;

            sample[p][0][-1] = sample[p][1][0  ];

            sample[p][1][ w] = sample[p][1][w-1];

            if (lbd)

                ret = encode_line(s, w, sample[p], (p + 1) / 2, 9);

            else

                ret = encode_line(s, w, sample[p], (p + 1) / 2, bits + 1);

            if (ret < 0)

                return ret;

        }

    }

    return 0;

}
