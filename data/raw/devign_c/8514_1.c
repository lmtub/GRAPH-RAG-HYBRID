static void GLZWDecodeInit(GifState * s, int csize)

{

    /* read buffer */

    s->eob_reached = 0;

    s->pbuf = s->buf;

    s->ebuf = s->buf;

    s->bbuf = 0;

    s->bbits = 0;



    /* decoder */

    s->codesize = csize;

    s->cursize = s->codesize + 1;

    s->curmask = mask[s->cursize];

    s->top_slot = 1 << s->cursize;

    s->clear_code = 1 << s->codesize;

    s->end_code = s->clear_code + 1;

    s->slot = s->newcodes = s->clear_code + 2;

    s->oc = s->fc = 0;

    s->sp = s->stack;

}
