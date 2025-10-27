static int get_cox(J2kDecoderContext *s, J2kCodingStyle *c)

{

    if (s->buf_end - s->buf < 5)

        return AVERROR(EINVAL);

          c->nreslevels = bytestream_get_byte(&s->buf) + 1; // num of resolution levels - 1

     c->log2_cblk_width = bytestream_get_byte(&s->buf) + 2; // cblk width

    c->log2_cblk_height = bytestream_get_byte(&s->buf) + 2; // cblk height



    c->cblk_style = bytestream_get_byte(&s->buf);

    if (c->cblk_style != 0){ // cblk style

        av_log(s->avctx, AV_LOG_WARNING, "extra cblk styles %X\n", c->cblk_style);

    }

    c->transform = bytestream_get_byte(&s->buf); // transformation

    if (c->csty & J2K_CSTY_PREC) {

        int i;

        for (i = 0; i < c->nreslevels; i++)

            bytestream_get_byte(&s->buf);

    }

    return 0;

}
