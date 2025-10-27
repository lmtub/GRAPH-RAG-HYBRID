static int h261_decode_gob_header(H261Context *h)

{

    unsigned int val;

    MpegEncContext *const s = &h->s;



    if (!h->gob_start_code_skipped) {

        /* Check for GOB Start Code */

        val = show_bits(&s->gb, 15);

        if (val)

            return -1;



        /* We have a GBSC */

        skip_bits(&s->gb, 16);

    }



    h->gob_start_code_skipped = 0;



    h->gob_number = get_bits(&s->gb, 4); /* GN */

    s->qscale     = get_bits(&s->gb, 5); /* GQUANT */



    /* Check if gob_number is valid */

    if (s->mb_height == 18) { // CIF

        if ((h->gob_number <= 0) || (h->gob_number > 12))

            return -1;

    } else { // QCIF

        if ((h->gob_number != 1) && (h->gob_number != 3) &&

            (h->gob_number != 5))

            return -1;

    }



    /* GEI */

    while (get_bits1(&s->gb) != 0)

        skip_bits(&s->gb, 8);



    if (s->qscale == 0) {

        av_log(s->avctx, AV_LOG_ERROR, "qscale has forbidden 0 value\n");

        if (s->avctx->err_recognition & (AV_EF_BITSTREAM | AV_EF_COMPLIANT))

            return -1;

    }



    /* For the first transmitted macroblock in a GOB, MBA is the absolute

     * address. For subsequent macroblocks, MBA is the difference between

     * the absolute addresses of the macroblock and the last transmitted

     * macroblock. */

    h->current_mba = 0;

    h->mba_diff    = 0;



    return 0;

}
