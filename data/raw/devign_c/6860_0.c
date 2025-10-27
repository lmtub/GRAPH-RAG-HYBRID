static void tile_codeblocks(Jpeg2000DecoderContext *s, Jpeg2000Tile *tile)

{

    Jpeg2000T1Context t1;



    int compno, reslevelno, bandno;



    /* Loop on tile components */



    for (compno = 0; compno < s->ncomponents; compno++) {

        Jpeg2000Component *comp     = tile->comp + compno;

        Jpeg2000CodingStyle *codsty = tile->codsty + compno;

        /* Loop on resolution levels */

        for (reslevelno = 0; reslevelno < codsty->nreslevels2decode; reslevelno++) {

            Jpeg2000ResLevel *rlevel = comp->reslevel + reslevelno;

            /* Loop on bands */

            for (bandno = 0; bandno < rlevel->nbands; bandno++) {

                uint16_t nb_precincts, precno;

                Jpeg2000Band *band = rlevel->band + bandno;

                int cblkno = 0, bandpos;

                bandpos = bandno + (reslevelno > 0);



                if (band->coord[0][0] == band->coord[0][1] ||

                    band->coord[1][0] == band->coord[1][1])

                    continue;



                nb_precincts = rlevel->num_precincts_x * rlevel->num_precincts_y;

                /* Loop on precincts */

                for (precno = 0; precno < nb_precincts; precno++) {

                    Jpeg2000Prec *prec = band->prec + precno;



                    /* Loop on codeblocks */

                    for (cblkno = 0; cblkno < prec->nb_codeblocks_width * prec->nb_codeblocks_height; cblkno++) {

                        int x, y;

                        Jpeg2000Cblk *cblk = prec->cblk + cblkno;

                        decode_cblk(s, codsty, &t1, cblk,

                                    cblk->coord[0][1] - cblk->coord[0][0],

                                    cblk->coord[1][1] - cblk->coord[1][0],

                                    bandpos);



                        x = cblk->coord[0][0];

                        y = cblk->coord[1][0];



                        if (codsty->transform == FF_DWT97)

                            dequantization_float(x, y, cblk, comp, &t1, band);

                        else

                            dequantization_int(x, y, cblk, comp, &t1, band);

                   } /* end cblk */

                } /*end prec */

            } /* end band */

        } /* end reslevel */



        /* inverse DWT */

        ff_dwt_decode(&comp->dwt, codsty->transform == FF_DWT97 ? (void*)comp->f_data : (void*)comp->i_data);

    } /*end comp */

}
