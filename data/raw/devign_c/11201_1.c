static int get_qcc(J2kDecoderContext *s, int n, J2kQuantStyle *q, uint8_t *properties)

{

    int compno;



    if (s->buf_end - s->buf < 1)

        return AVERROR(EINVAL);



    compno = bytestream_get_byte(&s->buf);

    properties[compno] |= HAD_QCC;

    return get_qcx(s, n-1, q+compno);

}
