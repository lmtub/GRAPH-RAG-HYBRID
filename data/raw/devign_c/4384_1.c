static int get_cod(J2kDecoderContext *s, J2kCodingStyle *c, uint8_t *properties)

{

    J2kCodingStyle tmp;

    int compno;



    if (s->buf_end - s->buf < 5)

        return AVERROR(EINVAL);



    tmp.log2_prec_width  =

    tmp.log2_prec_height = 15;



    tmp.csty = bytestream_get_byte(&s->buf);



    if (bytestream_get_byte(&s->buf)){ // progression level

        av_log(s->avctx, AV_LOG_ERROR, "only LRCP progression supported\n");

        return -1;

    }



    tmp.nlayers = bytestream_get_be16(&s->buf);

        tmp.mct = bytestream_get_byte(&s->buf); // multiple component transformation



    get_cox(s, &tmp);

    for (compno = 0; compno < s->ncomponents; compno++){

        if (!(properties[compno] & HAD_COC))

            memcpy(c + compno, &tmp, sizeof(J2kCodingStyle));

    }

    return 0;

}
