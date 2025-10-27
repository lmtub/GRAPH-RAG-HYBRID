static int ivi_decode_blocks(GetBitContext *gb, IVIBandDesc *band,

                             IVITile *tile, AVCodecContext *avctx)

{

    int mbn, blk, num_blocks, blk_size, ret, is_intra, mc_type = 0;

    int mv_x = 0, mv_y = 0;

    int32_t prev_dc;

    uint32_t cbp, quant, buf_offs;

    IVIMbInfo *mb;

    ivi_mc_func mc_with_delta_func, mc_no_delta_func;

    const uint8_t *scale_tab;



    /* init intra prediction for the DC coefficient */

    prev_dc    = 0;

    blk_size   = band->blk_size;

    /* number of blocks per mb */

    num_blocks = (band->mb_size != blk_size) ? 4 : 1;

    if (blk_size == 8) {

        mc_with_delta_func = ff_ivi_mc_8x8_delta;

        mc_no_delta_func   = ff_ivi_mc_8x8_no_delta;

    } else {

        mc_with_delta_func = ff_ivi_mc_4x4_delta;

        mc_no_delta_func   = ff_ivi_mc_4x4_no_delta;

    }



    for (mbn = 0, mb = tile->mbs; mbn < tile->num_MBs; mb++, mbn++) {

        is_intra = !mb->type;

        cbp      = mb->cbp;

        buf_offs = mb->buf_offs;



        quant = band->glob_quant + mb->q_delta;

        if (avctx->codec_id == AV_CODEC_ID_INDEO4)

            quant = av_clip(quant, 0, 31);

        else

            quant = av_clip(quant, 0, 23);



        scale_tab = is_intra ? band->intra_scale : band->inter_scale;

        if (scale_tab)

            quant = scale_tab[quant];



        if (!is_intra) {

            mv_x = mb->mv_x;

            mv_y = mb->mv_y;

            if (band->is_halfpel) {

                mc_type = ((mv_y & 1) << 1) | (mv_x & 1);

                mv_x >>= 1;

                mv_y >>= 1; /* convert halfpel vectors into fullpel ones */

            }

            if (mb->type) {

                int dmv_x, dmv_y, cx, cy;



                dmv_x = mb->mv_x >> band->is_halfpel;

                dmv_y = mb->mv_y >> band->is_halfpel;

                cx    = mb->mv_x &  band->is_halfpel;

                cy    = mb->mv_y &  band->is_halfpel;



                if (mb->xpos + dmv_x < 0 ||

                    mb->xpos + dmv_x + band->mb_size + cx > band->pitch ||

                    mb->ypos + dmv_y < 0 ||

                    mb->ypos + dmv_y + band->mb_size + cy > band->aheight) {

                    return AVERROR_INVALIDDATA;

                }

            }

        }



        for (blk = 0; blk < num_blocks; blk++) {

            /* adjust block position in the buffer according to its number */

            if (blk & 1) {

                buf_offs += blk_size;

            } else if (blk == 2) {

                buf_offs -= blk_size;

                buf_offs += blk_size * band->pitch;

            }



            if (cbp & 1) { /* block coded ? */

                ret = ivi_decode_coded_blocks(gb, band, mc_with_delta_func,

                                              mv_x, mv_y, &prev_dc, is_intra,

                                              mc_type, quant, buf_offs, avctx);

                if (ret < 0)

                    return ret;

            } else {

                /* block not coded */

                /* for intra blocks apply the dc slant transform */

                /* for inter - perform the motion compensation without delta */

                if (is_intra) {

                    if (band->dc_transform)

                        band->dc_transform(&prev_dc, band->buf + buf_offs,

                                           band->pitch, blk_size);

                } else {

                    ret = ivi_mc(mc_no_delta_func, band->buf, band->ref_buf,

                                 buf_offs, mv_x, mv_y, band->pitch, mc_type);

                    if (ret < 0)

                        return ret;

                }

            }



            cbp >>= 1;

        }// for blk

    }// for mbn



    align_get_bits(gb);



    return 0;

}
