static int ivi_decode_coded_blocks(GetBitContext *gb, IVIBandDesc *band,

                                   ivi_mc_func mc, int mv_x, int mv_y,

                                   int *prev_dc, int is_intra, int mc_type,

                                   uint32_t quant, int offs,

                                   AVCodecContext *avctx)

{

    const uint16_t *base_tab  = is_intra ? band->intra_base : band->inter_base;

    RVMapDesc *rvmap = band->rv_map;

    uint8_t col_flags[8];

    int32_t trvec[64];

    uint32_t sym = 0, lo, hi, q;

    int pos, run, val;

    int blk_size   = band->blk_size;

    int num_coeffs = blk_size * blk_size;

    int col_mask   = blk_size - 1;

    int scan_pos   = -1;

    int min_size   = band->pitch * (band->transform_size - 1) +

                     band->transform_size;

    int buf_size   = band->pitch * band->aheight - offs;



    if (min_size > buf_size)

        return AVERROR_INVALIDDATA;



    if (!band->scan) {

        av_log(avctx, AV_LOG_ERROR, "Scan pattern is not set.\n");

        return AVERROR_INVALIDDATA;

    }



    /* zero transform vector */

    memset(trvec, 0, num_coeffs * sizeof(trvec[0]));

    /* zero column flags */

    memset(col_flags, 0, sizeof(col_flags));

    while (scan_pos <= num_coeffs) {

        sym = get_vlc2(gb, band->blk_vlc.tab->table,

                       IVI_VLC_BITS, 1);

        if (sym == rvmap->eob_sym)

            break; /* End of block */



        /* Escape - run/val explicitly coded using 3 vlc codes */

        if (sym == rvmap->esc_sym) {

            run = get_vlc2(gb, band->blk_vlc.tab->table, IVI_VLC_BITS, 1) + 1;

            lo  = get_vlc2(gb, band->blk_vlc.tab->table, IVI_VLC_BITS, 1);

            hi  = get_vlc2(gb, band->blk_vlc.tab->table, IVI_VLC_BITS, 1);

            /* merge them and convert into signed val */

            val = IVI_TOSIGNED((hi << 6) | lo);

        } else {

            if (sym >= 256U) {

                av_log(avctx, AV_LOG_ERROR, "Invalid sym encountered: %d.\n", sym);

                return AVERROR_INVALIDDATA;

            }

            run = rvmap->runtab[sym];

            val = rvmap->valtab[sym];

        }



        /* de-zigzag and dequantize */

        scan_pos += run;

        if (scan_pos >= num_coeffs || scan_pos < 0)

            break;

        pos = band->scan[scan_pos];



        if (!val)

            av_dlog(avctx, "Val = 0 encountered!\n");



        q = (base_tab[pos] * quant) >> 9;

        if (q > 1)

            val = val * q + FFSIGN(val) * (((q ^ 1) - 1) >> 1);

        trvec[pos] = val;

        /* track columns containing non-zero coeffs */

        col_flags[pos & col_mask] |= !!val;

    }



    if (scan_pos < 0 || scan_pos >= num_coeffs && sym != rvmap->eob_sym)

        return AVERROR_INVALIDDATA; /* corrupt block data */



    /* undoing DC coeff prediction for intra-blocks */

    if (is_intra && band->is_2d_trans) {

        *prev_dc     += trvec[0];

        trvec[0]      = *prev_dc;

        col_flags[0] |= !!*prev_dc;

    }



    /* apply inverse transform */

    band->inv_transform(trvec, band->buf + offs,

                        band->pitch, col_flags);



    /* apply motion compensation */

    if (!is_intra)

        return ivi_mc(mc, band->buf, band->ref_buf, offs, mv_x, mv_y,

                      band->pitch, mc_type);



    return 0;

}
