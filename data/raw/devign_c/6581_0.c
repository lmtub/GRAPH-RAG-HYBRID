static int gif_parse_next_image(GifState *s)

{

    for (;;) {

        int code = bytestream_get_byte(&s->bytestream);

#ifdef DEBUG

        dprintf(s->avctx, "gif: code=%02x '%c'\n", code, code);

#endif

        switch (code) {

        case ',':

            if (gif_read_image(s) < 0)

                return -1;

            return 0;

        case ';':

            /* end of image */

            return -1;

        case '!':

            if (gif_read_extension(s) < 0)

                return -1;

            break;

        default:

            /* error or errneous EOF */

            return -1;

        }

    }

}
