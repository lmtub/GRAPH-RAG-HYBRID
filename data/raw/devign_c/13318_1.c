static inline int GetCode(GifState * s)

{

    int c, sizbuf;

    uint8_t *ptr;



    while (s->bbits < s->cursize) {

        ptr = s->pbuf;

        if (ptr >= s->ebuf) {

            if (!s->eob_reached) {

                sizbuf = get_byte(s->f);

                s->ebuf = s->buf + sizbuf;

                s->pbuf = s->buf;

                if (sizbuf > 0) {

                    get_buffer(s->f, s->buf, sizbuf);

                } else {

                    s->eob_reached = 1;

                }

            }

            ptr = s->pbuf;

        }

        s->bbuf |= ptr[0] << s->bbits;

        ptr++;

        s->pbuf = ptr;

        s->bbits += 8;

    }

    c = s->bbuf & s->curmask;

    s->bbuf >>= s->cursize;

    s->bbits -= s->cursize;

    return c;

}
