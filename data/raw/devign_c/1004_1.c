static int GLZWDecode(GifState * s, uint8_t * buf, int len)

{

    int l, c, code, oc, fc;

    uint8_t *sp;



    if (s->end_code < 0)

        return 0;



    l = len;

    sp = s->sp;

    oc = s->oc;

    fc = s->fc;



    while (sp > s->stack) {

        *buf++ = *(--sp);

        if ((--l) == 0)

            goto the_end;

    }



    for (;;) {

        c = GetCode(s);

        if (c == s->end_code) {

            s->end_code = -1;

            break;

        } else if (c == s->clear_code) {

            s->cursize = s->codesize + 1;

            s->curmask = mask[s->cursize];

            s->slot = s->newcodes;

            s->top_slot = 1 << s->cursize;

            while ((c = GetCode(s)) == s->clear_code);

            if (c == s->end_code) {

                s->end_code = -1;

                break;

            }

            /* test error */

            if (c >= s->slot)

                c = 0;

            fc = oc = c;

            *buf++ = c;

            if ((--l) == 0)

                break;

        } else {

            code = c;

            if (code >= s->slot) {

                *sp++ = fc;

                code = oc;

            }

            while (code >= s->newcodes) {

                *sp++ = s->suffix[code];

                code = s->prefix[code];

            }

            *sp++ = code;

            if (s->slot < s->top_slot) {

                s->suffix[s->slot] = fc = code;

                s->prefix[s->slot++] = oc;

                oc = c;

            }

            if (s->slot >= s->top_slot) {

                if (s->cursize < MAXBITS) {

                    s->top_slot <<= 1;

                    s->curmask = mask[++s->cursize];

                }

            }

            while (sp > s->stack) {

                *buf++ = *(--sp);

                if ((--l) == 0)

                    goto the_end;

            }

        }

    }

  the_end:

    s->sp = sp;

    s->oc = oc;

    s->fc = fc;

    return len - l;

}
