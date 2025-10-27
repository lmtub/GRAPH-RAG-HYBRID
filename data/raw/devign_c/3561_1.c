static int jpeg2000_decode_packet(Jpeg2000DecoderContext *s, Jpeg2000Tile *tile, int *tp_index,

                                  Jpeg2000CodingStyle *codsty,

                                  Jpeg2000ResLevel *rlevel, int precno,

                                  int layno, uint8_t *expn, int numgbits)

{

    int bandno, cblkno, ret, nb_code_blocks;

    int cwsno;



    if (layno < rlevel->band[0].prec[precno].decoded_layers)

        return 0;

    rlevel->band[0].prec[precno].decoded_layers = layno + 1;



    if (bytestream2_get_bytes_left(&s->g) == 0 && s->bit_index == 8) {

        if (*tp_index < FF_ARRAY_ELEMS(tile->tile_part) - 1) {

            s->g = tile->tile_part[++(*tp_index)].tpg;

        }

    }



    if (bytestream2_peek_be32(&s->g) == JPEG2000_SOP_FIXED_BYTES)

        bytestream2_skip(&s->g, JPEG2000_SOP_BYTE_LENGTH);



    if (!(ret = get_bits(s, 1))) {

        jpeg2000_flush(s);

        return 0;

    } else if (ret < 0)

        return ret;



    for (bandno = 0; bandno < rlevel->nbands; bandno++) {

        Jpeg2000Band *band = rlevel->band + bandno;

        Jpeg2000Prec *prec = band->prec + precno;



        if (band->coord[0][0] == band->coord[0][1] ||

            band->coord[1][0] == band->coord[1][1])

            continue;

        nb_code_blocks =  prec->nb_codeblocks_height *

                          prec->nb_codeblocks_width;

        for (cblkno = 0; cblkno < nb_code_blocks; cblkno++) {

            Jpeg2000Cblk *cblk = prec->cblk + cblkno;

            int incl, newpasses, llen;



            if (cblk->npasses)

                incl = get_bits(s, 1);

            else

                incl = tag_tree_decode(s, prec->cblkincl + cblkno, layno + 1) == layno;

            if (!incl)

                continue;

            else if (incl < 0)

                return incl;



            if (!cblk->npasses) {

                int v = expn[bandno] + numgbits - 1 -

                        tag_tree_decode(s, prec->zerobits + cblkno, 100);

                if (v < 0 || v > 30) {

                    av_log(s->avctx, AV_LOG_ERROR,

                           "nonzerobits %d invalid or unsupported\n", v);

                    return AVERROR_INVALIDDATA;

                }

                cblk->nonzerobits = v;

            }

            if ((newpasses = getnpasses(s)) < 0)

                return newpasses;

            av_assert2(newpasses > 0);

            if (cblk->npasses + newpasses >= JPEG2000_MAX_PASSES) {

                avpriv_request_sample(s->avctx, "Too many passes");

                return AVERROR_PATCHWELCOME;

            }

            if ((llen = getlblockinc(s)) < 0)

                return llen;

            if (cblk->lblock + llen + av_log2(newpasses) > 16) {

                avpriv_request_sample(s->avctx,

                                      "Block with length beyond 16 bits");

                return AVERROR_PATCHWELCOME;

            }



            cblk->lblock += llen;



            cblk->nb_lengthinc = 0;

            cblk->nb_terminationsinc = 0;

            do {

                int newpasses1 = 0;



                while (newpasses1 < newpasses) {

                    newpasses1 ++;

                    if (needs_termination(codsty->cblk_style, cblk->npasses + newpasses1 - 1)) {

                        cblk->nb_terminationsinc ++;

                        break;

                    }

                }



                if ((ret = get_bits(s, av_log2(newpasses1) + cblk->lblock)) < 0)

                    return ret;

                if (ret > sizeof(cblk->data)) {

                    avpriv_request_sample(s->avctx,

                                        "Block with lengthinc greater than %"SIZE_SPECIFIER"",

                                        sizeof(cblk->data));

                    return AVERROR_PATCHWELCOME;

                }

                cblk->lengthinc[cblk->nb_lengthinc++] = ret;

                cblk->npasses  += newpasses1;

                newpasses -= newpasses1;

            } while(newpasses);

        }

    }

    jpeg2000_flush(s);



    if (codsty->csty & JPEG2000_CSTY_EPH) {

        if (bytestream2_peek_be16(&s->g) == JPEG2000_EPH)

            bytestream2_skip(&s->g, 2);

        else

            av_log(s->avctx, AV_LOG_ERROR, "EPH marker not found. instead %X\n", bytestream2_peek_be32(&s->g));

    }



    for (bandno = 0; bandno < rlevel->nbands; bandno++) {

        Jpeg2000Band *band = rlevel->band + bandno;

        Jpeg2000Prec *prec = band->prec + precno;



        nb_code_blocks = prec->nb_codeblocks_height * prec->nb_codeblocks_width;

        for (cblkno = 0; cblkno < nb_code_blocks; cblkno++) {

            Jpeg2000Cblk *cblk = prec->cblk + cblkno;

            for (cwsno = 0; cwsno < cblk->nb_lengthinc; cwsno ++) {

                if (   bytestream2_get_bytes_left(&s->g) < cblk->lengthinc[cwsno]

                    || sizeof(cblk->data) < cblk->length + cblk->lengthinc[cwsno] + 4

                ) {

                    av_log(s->avctx, AV_LOG_ERROR,

                        "Block length %"PRIu16" or lengthinc %d is too large, left %d\n",

                        cblk->length, cblk->lengthinc[cwsno], bytestream2_get_bytes_left(&s->g));

                    return AVERROR_INVALIDDATA;

                }



                bytestream2_get_bufferu(&s->g, cblk->data + cblk->length, cblk->lengthinc[cwsno]);

                cblk->length   += cblk->lengthinc[cwsno];

                cblk->lengthinc[cwsno] = 0;

                if (cblk->nb_terminationsinc) {

                    cblk->nb_terminationsinc--;

                    cblk->nb_terminations++;

                    cblk->data[cblk->length++] = 0xFF;

                    cblk->data[cblk->length++] = 0xFF;

                    cblk->data_start[cblk->nb_terminations] = cblk->length;

                }

            }

        }

    }

    return 0;

}
