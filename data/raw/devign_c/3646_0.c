static void encode_plane(FFV1Context *s, uint8_t *src, int w, int h,

                         int stride, int plane_index)

{

    int x, y, i;

    const int ring_size = s->avctx->context_model ? 3 : 2;

    int16_t *sample[3];

    s->run_index = 0;



    memset(s->sample_buffer, 0, ring_size * (w + 6) * sizeof(*s->sample_buffer));



    for (y = 0; y < h; y++) {

        for (i = 0; i < ring_size; i++)

            sample[i] = s->sample_buffer + (w + 6) * ((h + i - y) % ring_size) + 3;



        sample[0][-1]= sample[1][0  ];

        sample[1][ w]= sample[1][w-1];

// { START_TIMER

        if (s->bits_per_raw_sample <= 8) {

            for (x = 0; x < w; x++)

                sample[0][x] = src[x + stride * y];

            encode_line(s, w, sample, plane_index, 8);

        } else {

            if (s->packed_at_lsb) {

                for (x = 0; x < w; x++) {

                    sample[0][x] = ((uint16_t*)(src + stride*y))[x];

                }

            } else {

                for (x = 0; x < w; x++) {

                    sample[0][x] = ((uint16_t*)(src + stride*y))[x] >> (16 - s->bits_per_raw_sample);

                }

            }

            encode_line(s, w, sample, plane_index, s->bits_per_raw_sample);

        }

// STOP_TIMER("encode line") }

    }

}
