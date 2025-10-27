static av_always_inline void decode_bgr_1(HYuvContext *s, int count,

                                          int decorrelate, int alpha)

{

    int i;

    OPEN_READER(re, &s->gb);



    for (i = 0; i < count && get_bits_left(&s->gb) > 0; i++) {

        unsigned int index;

        int code, n;



        UPDATE_CACHE(re, &s->gb);

        index = SHOW_UBITS(re, &s->gb, VLC_BITS);

        n     = s->vlc[4].table[index][1];



        if (n>0) {

            code  = s->vlc[4].table[index][0];

            *(uint32_t*)&s->temp[0][4 * i] = s->pix_bgr_map[code];

            LAST_SKIP_BITS(re, &s->gb, n);

        } else {

            int nb_bits;

            if(decorrelate) {

                VLC_INTERN(s->temp[0][4 * i + G], s->vlc[1].table,

                           &s->gb, re, VLC_BITS, 3);



                UPDATE_CACHE(re, &s->gb);

                index = SHOW_UBITS(re, &s->gb, VLC_BITS);

                VLC_INTERN(code, s->vlc[0].table, &s->gb, re, VLC_BITS, 3);

                s->temp[0][4 * i + B] = code + s->temp[0][4 * i + G];



                UPDATE_CACHE(re, &s->gb);

                index = SHOW_UBITS(re, &s->gb, VLC_BITS);

                VLC_INTERN(code, s->vlc[2].table, &s->gb, re, VLC_BITS, 3);

                s->temp[0][4 * i + R] = code + s->temp[0][4 * i + G];

            } else {

                VLC_INTERN(s->temp[0][4 * i + B], s->vlc[0].table,

                           &s->gb, re, VLC_BITS, 3);



                UPDATE_CACHE(re, &s->gb);

                index = SHOW_UBITS(re, &s->gb, VLC_BITS);

                VLC_INTERN(s->temp[0][4 * i + G], s->vlc[1].table,

                           &s->gb, re, VLC_BITS, 3);



                UPDATE_CACHE(re, &s->gb);

                index = SHOW_UBITS(re, &s->gb, VLC_BITS);

                VLC_INTERN(s->temp[0][4 * i + R], s->vlc[2].table,

                           &s->gb, re, VLC_BITS, 3);

            }

            if (alpha) {

                UPDATE_CACHE(re, &s->gb);

                index = SHOW_UBITS(re, &s->gb, VLC_BITS);

                VLC_INTERN(s->temp[0][4 * i + A], s->vlc[2].table,

                           &s->gb, re, VLC_BITS, 3);

            }

        }

    }

    CLOSE_READER(re, &s->gb);

}
