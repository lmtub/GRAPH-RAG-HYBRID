static void decode_plane_bitstream(HYuvContext *s, int count, int plane)

{

    int i;



    count /= 2;



    if (s->bps <= 8) {

        OPEN_READER(re, &s->gb);

        if (count >= (get_bits_left(&s->gb)) / (32 * 2)) {

            for (i = 0; i < count && get_bits_left(&s->gb) > 0; i++) {

                READ_2PIX_PLANE(s->temp[0][2 * i], s->temp[0][2 * i + 1], plane, OP8bits);

            }

        } else {

            for(i=0; i<count; i++){

                READ_2PIX_PLANE(s->temp[0][2 * i], s->temp[0][2 * i + 1], plane, OP8bits);

            }

        }

        CLOSE_READER(re, &s->gb);

    } else if (s->bps <= 14) {

        OPEN_READER(re, &s->gb);

        if (count >= (get_bits_left(&s->gb)) / (32 * 2)) {

            for (i = 0; i < count && get_bits_left(&s->gb) > 0; i++) {

                READ_2PIX_PLANE(s->temp16[0][2 * i], s->temp16[0][2 * i + 1], plane, OP14bits);

            }

        } else {

            for(i=0; i<count; i++){

                READ_2PIX_PLANE(s->temp16[0][2 * i], s->temp16[0][2 * i + 1], plane, OP14bits);

            }

        }

        CLOSE_READER(re, &s->gb);

    } else {

        if (count >= (get_bits_left(&s->gb)) / (32 * 2)) {

            for (i = 0; i < count && get_bits_left(&s->gb) > 0; i++) {

                READ_2PIX_PLANE16(s->temp16[0][2 * i], s->temp16[0][2 * i + 1], plane);

            }

        } else {

            for(i=0; i<count; i++){

                READ_2PIX_PLANE16(s->temp16[0][2 * i], s->temp16[0][2 * i + 1], plane);

            }

        }

    }

}
