static int get_cod(Jpeg2000DecoderContext *s, Jpeg2000CodingStyle *c,

                   uint8_t *properties)

{

    Jpeg2000CodingStyle tmp;

    int compno;



    if (s->buf_end - s->buf < 5)

        return AVERROR_INVALIDDATA;



    tmp.log2_prec_width  =

    tmp.log2_prec_height = 15;



    tmp.csty = bytestream_get_byte(&s->buf);



    // get progression order

    tmp.prog_order = bytestream_get_byte(&s->buf);



    tmp.nlayers = bytestream_get_be16(&s->buf);

    tmp.mct     = bytestream_get_byte(&s->buf); // multiple component transformation



    get_cox(s, &tmp);

    for (compno = 0; compno < s->ncomponents; compno++)

        if (!(properties[compno] & HAD_COC))

            memcpy(c + compno, &tmp, sizeof(tmp));

    return 0;

}
