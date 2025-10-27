static int h261_decode_gob_header(H261Context *h){

    unsigned int val;

    MpegEncContext * const s = &h->s;

    

    /* Check for GOB Start Code */

    val = show_bits(&s->gb, 15);

    if(val)

        return -1;



    /* We have a GBSC */

    skip_bits(&s->gb, 16);



    h->gob_number = get_bits(&s->gb, 4); /* GN */

    s->qscale = get_bits(&s->gb, 5); /* GQUANT */



    /* GEI */

    while (get_bits1(&s->gb) != 0) {

        skip_bits(&s->gb, 8);

    }



    if(s->qscale==0)

        return -1;



    // For the first transmitted macroblock in a GOB, MBA is the absolute address. For

    // subsequent macroblocks, MBA is the difference between the absolute addresses of

    // the macroblock and the last transmitted macroblock.

    h->current_mba = 0;

    h->mba_diff = 0;



    return 0;

}
