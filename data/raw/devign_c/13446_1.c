static int jpeg2000_read_main_headers(Jpeg2000DecoderContext *s)

{

    Jpeg2000CodingStyle *codsty = s->codsty;

    Jpeg2000QuantStyle *qntsty  = s->qntsty;

    uint8_t *properties         = s->properties;



    for (;;) {

        int len, ret = 0;

        uint16_t marker;

        int oldpos;



        if (bytestream2_get_bytes_left(&s->g) < 2) {

            av_log(s->avctx, AV_LOG_ERROR, "Missing EOC\n");

            break;

        }



        marker = bytestream2_get_be16u(&s->g);

        oldpos = bytestream2_tell(&s->g);



        if (marker == JPEG2000_EOC)

            break;



        if (bytestream2_get_bytes_left(&s->g) < 2)

            return AVERROR_INVALIDDATA;

        len = bytestream2_get_be16u(&s->g);

        switch (marker) {

        case JPEG2000_SIZ:

            ret = get_siz(s);

            break;

        case JPEG2000_COC:

            ret = get_coc(s, codsty, properties);

            break;

        case JPEG2000_COD:

            ret = get_cod(s, codsty, properties);

            break;

        case JPEG2000_QCC:

            ret = get_qcc(s, len, qntsty, properties);

            break;

        case JPEG2000_QCD:

            ret = get_qcd(s, len, qntsty, properties);

            break;

        case JPEG2000_SOT:

            ret = get_sot(s, len);

            break;

        case JPEG2000_COM:

            // the comment is ignored

            bytestream2_skip(&s->g, len - 2);

            break;

        case JPEG2000_TLM:

            // Tile-part lengths

            ret = get_tlm(s, len);

            break;

        default:

            av_log(s->avctx, AV_LOG_ERROR,

                   "unsupported marker 0x%.4X at pos 0x%X\n",

                   marker, bytestream2_tell(&s->g) - 4);

            bytestream2_skip(&s->g, len - 2);

            break;

        }

        if (((bytestream2_tell(&s->g) - oldpos != len) && (marker != JPEG2000_SOT)) || ret) {

            av_log(s->avctx, AV_LOG_ERROR,

                   "error during processing marker segment %.4x\n", marker);

            return ret ? ret : -1;

        }

    }

    return 0;

}
