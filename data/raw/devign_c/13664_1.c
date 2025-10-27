static int decode_tile(Jpeg2000DecoderContext *s, Jpeg2000Tile *tile)

{

    int compno, reslevelno, bandno;

    int x, y;

    uint8_t *line;

    Jpeg2000T1Context t1;



    /* Loop on tile components */

    for (compno = 0; compno < s->ncomponents; compno++) {

        Jpeg2000Component *comp     = tile->comp + compno;

        Jpeg2000CodingStyle *codsty = tile->codsty + compno;



        /* Loop on resolution levels */

        for (reslevelno = 0; reslevelno < codsty->nreslevels2decode; reslevelno++) {

            Jpeg2000ResLevel *rlevel = comp->reslevel + reslevelno;

            /* Loop on bands */

            for (bandno = 0; bandno < rlevel->nbands; bandno++) {

                int nb_precincts, precno;

                Jpeg2000Band *band = rlevel->band + bandno;

                int cblkno=0, bandpos;



                bandpos = bandno + (reslevelno > 0);



                if (band->coord[0][0] == band->coord[0][1] || band->coord[1][0] == band->coord[1][1])

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



                        /* Manage band offsets */

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



        ff_dwt_decode(&comp->dwt, comp->data);

    } /*end comp */



    /* inverse MCT transformation */

    if (tile->codsty[0].mct)

        mct_decode(s, tile);



    if (s->precision <= 8) {

        for (compno = 0; compno < s->ncomponents; compno++) {

            Jpeg2000Component *comp = tile->comp + compno;

            float *datap = (float*)comp->data;

            int32_t *i_datap = (int32_t *) comp->data;



            y = tile->comp[compno].coord[1][0] - s->image_offset_y;

            line = s->picture->data[0] + y * s->picture->linesize[0];

            for (; y < tile->comp[compno].coord[1][1] - s->image_offset_y; y += s->cdy[compno]) {

                uint8_t *dst;



                x   = tile->comp[compno].coord[0][0] - s->image_offset_x;

                dst = line + x * s->ncomponents + compno;



                for (; x < tile->comp[compno].coord[0][1] - s->image_offset_x; x += s->cdx[compno]) {

                     int val;

                    /* DC level shift and clip see ISO 15444-1:2002 G.1.2 */

                    if (tile->codsty->transform == FF_DWT97)

                        val = lrintf(*datap) + (1 << (s->cbps[compno] - 1));

                    else

                        val = *i_datap + (1 << (s->cbps[compno] - 1));

                    val = av_clip(val, 0, (1 << s->cbps[compno]) - 1);

                    *dst = val << (8 - s->cbps[compno]);

                    datap++;

                    i_datap++;

                    dst += s->ncomponents;

                }

                line += s->picture->linesize[0];

            }

        }

    } else {

        for (compno = 0; compno < s->ncomponents; compno++) {

            Jpeg2000Component *comp = tile->comp + compno;

            float *datap = (float*)comp->data;

            int32_t *i_datap = (int32_t *) comp->data;

            uint16_t *linel;



            y     = tile->comp[compno].coord[1][0] - s->image_offset_y;

            linel = (uint16_t*)s->picture->data[0] + y * (s->picture->linesize[0] >> 1);

            for (; y < tile->comp[compno].coord[1][1] - s->image_offset_y; y += s->cdy[compno]) {

                uint16_t *dst;



                x = tile->comp[compno].coord[0][0] - s->image_offset_x;

                dst = linel + (x * s->ncomponents + compno);

                for (; x < tile->comp[compno].coord[0][1] - s->image_offset_x; x += s-> cdx[compno]) {

                     int val;

                    /* DC level shift and clip see ISO 15444-1:2002 G.1.2 */

                    if (tile->codsty->transform == FF_DWT97)

                        val = lrintf(*datap) + (1 << (s->cbps[compno] - 1));

                    else

                        val = *i_datap + (1 << (s->cbps[compno] - 1));

                    val = av_clip(val, 0, (1 << s->cbps[compno]) - 1);

                    /* align 12 bit values in little-endian mode */

                    *dst = val << (16 - s->cbps[compno]);

                    datap++;

                    i_datap++;

                    dst += s->ncomponents;

                }

                linel += s->picture->linesize[0]>>1;

            }

        }

    }

    return 0;

}
