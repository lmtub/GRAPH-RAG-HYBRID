static int get_coc(J2kDecoderContext *s, J2kCodingStyle *c, uint8_t *properties)

{

    int compno;



    if (s->buf_end - s->buf < 2)

        return AVERROR(EINVAL);



    compno = bytestream_get_byte(&s->buf);



    c += compno;

    c->csty = bytestream_get_byte(&s->buf);

    get_cox(s, c);



    properties[compno] |= HAD_COC;

    return 0;

}
