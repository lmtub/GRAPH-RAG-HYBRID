static int get_cox(Jpeg2000DecoderContext *s, Jpeg2000CodingStyle *c)

{

    uint8_t byte;



    if (s->buf_end - s->buf < 5)

        return AVERROR(EINVAL);

    c->nreslevels = bytestream_get_byte(&s->buf) + 1; // num of resolution levels - 1



    /* compute number of resolution levels to decode */

    if (c->nreslevels < s->reduction_factor)

        c->nreslevels2decode = 1;

    else

        c->nreslevels2decode = c->nreslevels - s->reduction_factor;



    c->log2_cblk_width  = bytestream_get_byte(&s->buf) + 2; // cblk width

    c->log2_cblk_height = bytestream_get_byte(&s->buf) + 2; // cblk height



    c->cblk_style = bytestream_get_byte(&s->buf);

    if (c->cblk_style != 0) { // cblk style

        av_log(s->avctx, AV_LOG_ERROR, "no extra cblk styles supported\n");

        return -1;

    }

    c->transform = bytestream_get_byte(&s->buf); // DWT transformation type

    /* set integer 9/7 DWT in case of BITEXACT flag */

    if ((s->avctx->flags & CODEC_FLAG_BITEXACT) && (c->transform == FF_DWT97))

        c->transform = FF_DWT97_INT;



    if (c->csty & JPEG2000_CSTY_PREC) {

        int i;

        for (i = 0; i < c->nreslevels; i++) {

            byte = bytestream_get_byte(&s->buf);

            c->log2_prec_widths[i]  =  byte       & 0x0F;    // precinct PPx

            c->log2_prec_heights[i] = (byte >> 4) & 0x0F;    // precinct PPy

        }

    }

    return 0;

}
