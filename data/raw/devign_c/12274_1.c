static int get_qcc(Jpeg2000DecoderContext *s, int n, Jpeg2000QuantStyle *q,

                   uint8_t *properties)

{

    int compno;



    if (bytestream2_get_bytes_left(&s->g) < 1)

        return AVERROR_INVALIDDATA;



    compno              = bytestream2_get_byteu(&s->g);

    properties[compno] |= HAD_QCC;

    return get_qcx(s, n - 1, q + compno);

}
