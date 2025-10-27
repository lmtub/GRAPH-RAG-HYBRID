static int lzw_get_code(struct LZWState * s)

{

    int c;



    if(s->mode == FF_LZW_GIF) {

        while (s->bbits < s->cursize) {

            if (!s->bs) {

                s->bs = *s->pbuf++;

                if(!s->bs) {

                    s->eob_reached = 1;

                    break;

                }

            }

            s->bbuf |= (*s->pbuf++) << s->bbits;

            s->bbits += 8;

            s->bs--;

        }

        c = s->bbuf & s->curmask;

        s->bbuf >>= s->cursize;

    } else { // TIFF

        while (s->bbits < s->cursize) {

            if (s->pbuf >= s->ebuf) {

                s->eob_reached = 1;

            }

            s->bbuf = (s->bbuf << 8) | (*s->pbuf++);

            s->bbits += 8;

        }

        c = (s->bbuf >> (s->bbits - s->cursize)) & s->curmask;

    }

    s->bbits -= s->cursize;

    return c;

}
