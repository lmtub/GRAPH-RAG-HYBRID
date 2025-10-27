int ff_jpeg2000_init_component(Jpeg2000Component *comp,
                               Jpeg2000CodingStyle *codsty,
                               Jpeg2000QuantStyle *qntsty,
                               int cbps, int dx, int dy,
                               AVCodecContext *avctx)
{
    int reslevelno, bandno, gbandno = 0, ret, i, j;
    uint32_t csize;
    if (codsty->nreslevels2decode <= 0) {
        av_log(avctx, AV_LOG_ERROR, "nreslevels2decode %d invalid or uninitialized\n", codsty->nreslevels2decode);
        return AVERROR_INVALIDDATA;
    }
    if (ret = ff_jpeg2000_dwt_init(&comp->dwt, comp->coord,
                                   codsty->nreslevels2decode - 1,
                                   codsty->transform))
        return ret;
    if (av_image_check_size(comp->coord[0][1] - comp->coord[0][0],
                            comp->coord[1][1] - comp->coord[1][0], 0, avctx))
        return AVERROR_INVALIDDATA;
    csize = (comp->coord[0][1] - comp->coord[0][0]) *
            (comp->coord[1][1] - comp->coord[1][0]);
    if (comp->coord[0][1] - comp->coord[0][0] > 32768 ||
        comp->coord[1][1] - comp->coord[1][0] > 32768) {
        av_log(avctx, AV_LOG_ERROR, "component size too large\n");
        return AVERROR_PATCHWELCOME;
    }
    if (codsty->transform == FF_DWT97) {
        csize += AV_INPUT_BUFFER_PADDING_SIZE / sizeof(*comp->f_data);
        comp->i_data = NULL;
        comp->f_data = av_mallocz_array(csize, sizeof(*comp->f_data));
        if (!comp->f_data)
    } else {
        csize += AV_INPUT_BUFFER_PADDING_SIZE / sizeof(*comp->i_data);
        comp->f_data = NULL;
        comp->i_data = av_mallocz_array(csize, sizeof(*comp->i_data));
        if (!comp->i_data)
    }
    comp->reslevel = av_mallocz_array(codsty->nreslevels, sizeof(*comp->reslevel));
    if (!comp->reslevel)
    /* LOOP on resolution levels */
    for (reslevelno = 0; reslevelno < codsty->nreslevels; reslevelno++) {
        int declvl = codsty->nreslevels - reslevelno;    // N_L -r see  ISO/IEC 15444-1:2002 B.5
        Jpeg2000ResLevel *reslevel = comp->reslevel + reslevelno;
        /* Compute borders for each resolution level.
         * Computation of trx_0, trx_1, try_0 and try_1.
         * see ISO/IEC 15444-1:2002 eq. B.5 and B-14 */
        for (i = 0; i < 2; i++)
            for (j = 0; j < 2; j++)
                reslevel->coord[i][j] =
                    ff_jpeg2000_ceildivpow2(comp->coord_o[i][j], declvl - 1);
        // update precincts size: 2^n value
        reslevel->log2_prec_width  = codsty->log2_prec_widths[reslevelno];
        reslevel->log2_prec_height = codsty->log2_prec_heights[reslevelno];
        if (!reslevel->log2_prec_width || !reslevel->log2_prec_height) {
            return AVERROR_INVALIDDATA;
        }
        /* Number of bands for each resolution level */
        if (reslevelno == 0)
            reslevel->nbands = 1;
        else
            reslevel->nbands = 3;
        /* Number of precincts which span the tile for resolution level reslevelno
         * see B.6 in ISO/IEC 15444-1:2002 eq. B-16
         * num_precincts_x = |- trx_1 / 2 ^ log2_prec_width) -| - (trx_0 / 2 ^ log2_prec_width)
         * num_precincts_y = |- try_1 / 2 ^ log2_prec_width) -| - (try_0 / 2 ^ log2_prec_width)
         * for Dcinema profiles in JPEG 2000
         * num_precincts_x = |- trx_1 / 2 ^ log2_prec_width) -|
         * num_precincts_y = |- try_1 / 2 ^ log2_prec_width) -| */
        if (reslevel->coord[0][1] == reslevel->coord[0][0])
            reslevel->num_precincts_x = 0;
        else
            reslevel->num_precincts_x =
                ff_jpeg2000_ceildivpow2(reslevel->coord[0][1],
                                        reslevel->log2_prec_width) -
                (reslevel->coord[0][0] >> reslevel->log2_prec_width);
        if (reslevel->coord[1][1] == reslevel->coord[1][0])
            reslevel->num_precincts_y = 0;
        else
            reslevel->num_precincts_y =
                ff_jpeg2000_ceildivpow2(reslevel->coord[1][1],
                                        reslevel->log2_prec_height) -
                (reslevel->coord[1][0] >> reslevel->log2_prec_height);
        reslevel->band = av_mallocz_array(reslevel->nbands, sizeof(*reslevel->band));
        if (!reslevel->band)
        for (bandno = 0; bandno < reslevel->nbands; bandno++, gbandno++) {
            ret = init_band(avctx, reslevel,
                            comp, codsty, qntsty,
                            bandno, gbandno, reslevelno,
                            cbps, dx, dy);
            if (ret < 0)
                return ret;
        }
    }
    return 0;
}