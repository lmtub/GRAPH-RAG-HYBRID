static void decode_422_bitstream(HYuvContext *s, int count)

{

    int i;



    count /= 2;



    if (count >= (get_bits_left(&s->gb)) / (31 * 4)) {

        for (i = 0; i < count && get_bits_left(&s->gb) > 0; i++) {

            READ_2PIX(s->temp[0][2 * i    ], s->temp[1][i], 1);

            READ_2PIX(s->temp[0][2 * i + 1], s->temp[2][i], 2);

        }

        for (; i < count; i++)

            s->temp[0][2 * i    ] = s->temp[1][i] =

            s->temp[0][2 * i + 1] = s->temp[2][i] = 128;

    } else {

        for (i = 0; i < count; i++) {

            READ_2PIX(s->temp[0][2 * i    ], s->temp[1][i], 1);

            READ_2PIX(s->temp[0][2 * i + 1], s->temp[2][i], 2);

        }

    }

}
