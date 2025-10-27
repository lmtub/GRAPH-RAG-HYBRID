static int cinepak_decode (CinepakContext *s)

{

    uint8_t      *eod = (s->data + s->size);

    int           i, result, strip_size, frame_flags, num_strips;

    int           y0 = 0;

    int           encoded_buf_size;

    /* if true, Cinepak data is from a Sega FILM/CPK file */

    int           sega_film_data = 0;



    if (s->size < 10)

        return -1;



    frame_flags = s->data[0];

    num_strips  = BE_16 (&s->data[8]);

    encoded_buf_size = ((s->data[1] << 16) | BE_16 (&s->data[2]));

    if (encoded_buf_size != s->size)

        sega_film_data = 1;

    if (sega_film_data)

        s->data    += 12;

    else

        s->data    += 10;



    if (num_strips > MAX_STRIPS)

        num_strips = MAX_STRIPS;



    for (i=0; i < num_strips; i++) {

        if ((s->data + 12) > eod)

            return -1;



        s->strips[i].id = BE_16 (s->data);

        s->strips[i].y1 = y0;

        s->strips[i].x1 = 0;

        s->strips[i].y2 = y0 + BE_16 (&s->data[8]);

        s->strips[i].x2 = s->avctx->width;



        strip_size = BE_16 (&s->data[2]) - 12;

        s->data   += 12;

        strip_size = ((s->data + strip_size) > eod) ? (eod - s->data) : strip_size;



        if ((i > 0) && !(frame_flags & 0x01)) {

            memcpy (s->strips[i].v4_codebook, s->strips[i-1].v4_codebook,

                sizeof(s->strips[i].v4_codebook));

            memcpy (s->strips[i].v1_codebook, s->strips[i-1].v1_codebook,

                sizeof(s->strips[i].v1_codebook));

        }



        result = cinepak_decode_strip (s, &s->strips[i], s->data, strip_size);



        if (result != 0)

            return result;



        s->data += strip_size;

        y0    = s->strips[i].y2;

    }

    return 0;

}
