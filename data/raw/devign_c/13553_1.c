static int init_prec(Jpeg2000Band *band,

                     Jpeg2000ResLevel *reslevel,

                     Jpeg2000Component *comp,

                     int precno, int bandno, int reslevelno,

                     int log2_band_prec_width,

                     int log2_band_prec_height)

{

    Jpeg2000Prec *prec = band->prec + precno;

    int nb_codeblocks, cblkno;



    prec->decoded_layers = 0;



    /* TODO: Explain formula for JPEG200 DCINEMA. */

    /* TODO: Verify with previous count of codeblocks per band */



    /* Compute P_x0 */

    prec->coord[0][0] = ((band->coord[0][0] >> log2_band_prec_width) + precno % reslevel->num_precincts_x) *

                        (1 << log2_band_prec_width);



    /* Compute P_y0 */

    prec->coord[1][0] = ((band->coord[1][0] >> log2_band_prec_height) + precno / reslevel->num_precincts_x) *

                        (1 << log2_band_prec_height);



    /* Compute P_x1 */

    prec->coord[0][1] = prec->coord[0][0] +

                        (1 << log2_band_prec_width);

    prec->coord[0][0] = FFMAX(prec->coord[0][0], band->coord[0][0]);

    prec->coord[0][1] = FFMIN(prec->coord[0][1], band->coord[0][1]);



    /* Compute P_y1 */

    prec->coord[1][1] = prec->coord[1][0] +

                        (1 << log2_band_prec_height);

    prec->coord[1][0] = FFMAX(prec->coord[1][0], band->coord[1][0]);

    prec->coord[1][1] = FFMIN(prec->coord[1][1], band->coord[1][1]);



    prec->nb_codeblocks_width =

        ff_jpeg2000_ceildivpow2(prec->coord[0][1],

                                band->log2_cblk_width)

        - (prec->coord[0][0] >> band->log2_cblk_width);

    prec->nb_codeblocks_height =

        ff_jpeg2000_ceildivpow2(prec->coord[1][1],

                                band->log2_cblk_height)

        - (prec->coord[1][0] >> band->log2_cblk_height);





    /* Tag trees initialization */

    prec->cblkincl =

        ff_jpeg2000_tag_tree_init(prec->nb_codeblocks_width,

                                  prec->nb_codeblocks_height);

    if (!prec->cblkincl)

        return AVERROR(ENOMEM);



    prec->zerobits =

        ff_jpeg2000_tag_tree_init(prec->nb_codeblocks_width,

                                  prec->nb_codeblocks_height);

    if (!prec->zerobits)

        return AVERROR(ENOMEM);



    if (prec->nb_codeblocks_width * (uint64_t)prec->nb_codeblocks_height > INT_MAX) {

        prec->cblk = NULL;

        return AVERROR(ENOMEM);

    }

    nb_codeblocks = prec->nb_codeblocks_width * prec->nb_codeblocks_height;

    prec->cblk = av_mallocz_array(nb_codeblocks, sizeof(*prec->cblk));

    if (!prec->cblk)

        return AVERROR(ENOMEM);

    for (cblkno = 0; cblkno < nb_codeblocks; cblkno++) {

        Jpeg2000Cblk *cblk = prec->cblk + cblkno;

        int Cx0, Cy0;



        /* Compute coordinates of codeblocks */

        /* Compute Cx0*/

        Cx0 = ((prec->coord[0][0]) >> band->log2_cblk_width) << band->log2_cblk_width;

        Cx0 = Cx0 + ((cblkno % prec->nb_codeblocks_width)  << band->log2_cblk_width);

        cblk->coord[0][0] = FFMAX(Cx0, prec->coord[0][0]);



        /* Compute Cy0*/

        Cy0 = ((prec->coord[1][0]) >> band->log2_cblk_height) << band->log2_cblk_height;

        Cy0 = Cy0 + ((cblkno / prec->nb_codeblocks_width)   << band->log2_cblk_height);

        cblk->coord[1][0] = FFMAX(Cy0, prec->coord[1][0]);



        /* Compute Cx1 */

        cblk->coord[0][1] = FFMIN(Cx0 + (1 << band->log2_cblk_width),

                                  prec->coord[0][1]);



        /* Compute Cy1 */

        cblk->coord[1][1] = FFMIN(Cy0 + (1 << band->log2_cblk_height),

                                  prec->coord[1][1]);

        /* Update code-blocks coordinates according sub-band position */

        if ((bandno + !!reslevelno) & 1) {

            cblk->coord[0][0] += comp->reslevel[reslevelno-1].coord[0][1] -

                                 comp->reslevel[reslevelno-1].coord[0][0];

            cblk->coord[0][1] += comp->reslevel[reslevelno-1].coord[0][1] -

                                 comp->reslevel[reslevelno-1].coord[0][0];

        }

        if ((bandno + !!reslevelno) & 2) {

            cblk->coord[1][0] += comp->reslevel[reslevelno-1].coord[1][1] -

                                 comp->reslevel[reslevelno-1].coord[1][0];

            cblk->coord[1][1] += comp->reslevel[reslevelno-1].coord[1][1] -

                                 comp->reslevel[reslevelno-1].coord[1][0];

        }



        cblk->zero      = 0;

        cblk->lblock    = 3;

        cblk->length    = 0;

        memset(cblk->lengthinc, 0, sizeof(cblk->lengthinc));

        cblk->npasses   = 0;

    }



    return 0;

}
