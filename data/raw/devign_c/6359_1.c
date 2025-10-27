static int decode_band(IVI5DecContext *ctx, int plane_num,

                       IVIBandDesc *band, AVCodecContext *avctx)

{

    int         result, i, t, idx1, idx2;

    IVITile     *tile;

    uint16_t    chksum;



    band->buf     = band->bufs[ctx->dst_buf];

    band->ref_buf = band->bufs[ctx->ref_buf];

    band->data_ptr = ctx->frame_data + (get_bits_count(&ctx->gb) >> 3);



    result = decode_band_hdr(ctx, band, avctx);

    if (result) {

        av_log(avctx, AV_LOG_ERROR, "Error while decoding band header: %d\n",

               result);

        return -1;

    }



    if (band->is_empty) {

        av_log(avctx, AV_LOG_ERROR, "Empty band encountered!\n");

        return -1;

    }



    band->rv_map = &ctx->rvmap_tabs[band->rvmap_sel];



    /* apply corrections to the selected rvmap table if present */

    for (i = 0; i < band->num_corr; i++) {

        idx1 = band->corr[i*2];

        idx2 = band->corr[i*2+1];

        FFSWAP(uint8_t, band->rv_map->runtab[idx1], band->rv_map->runtab[idx2]);

        FFSWAP(int16_t, band->rv_map->valtab[idx1], band->rv_map->valtab[idx2]);

    }



    for (t = 0; t < band->num_tiles; t++) {

        tile = &band->tiles[t];



        tile->is_empty = get_bits1(&ctx->gb);

        if (tile->is_empty) {

            ff_ivi_process_empty_tile(avctx, band, tile,

                                      (ctx->planes[0].bands[0].mb_size >> 3) - (band->mb_size >> 3));

            align_get_bits(&ctx->gb);

        } else {

            tile->data_size = ff_ivi_dec_tile_data_size(&ctx->gb);



            result = decode_mb_info(ctx, band, tile, avctx);

            if (result < 0)

                break;



            if (band->blk_size == 8) {

                band->intra_base  = &ivi5_base_quant_8x8_intra[band->quant_mat][0];

                band->inter_base  = &ivi5_base_quant_8x8_inter[band->quant_mat][0];

                band->intra_scale = &ivi5_scale_quant_8x8_intra[band->quant_mat][0];

                band->inter_scale = &ivi5_scale_quant_8x8_inter[band->quant_mat][0];

            } else {

                band->intra_base  = ivi5_base_quant_4x4_intra;

                band->inter_base  = ivi5_base_quant_4x4_inter;

                band->intra_scale = ivi5_scale_quant_4x4_intra;

                band->inter_scale = ivi5_scale_quant_4x4_inter;

            }



            result = ff_ivi_decode_blocks(&ctx->gb, band, tile);

            if (result < 0) {

                av_log(avctx, AV_LOG_ERROR, "Corrupted blocks data encountered!\n");

                break;

            }

        }

    }



    /* restore the selected rvmap table by applying its corrections in reverse order */

    for (i = band->num_corr-1; i >= 0; i--) {

        idx1 = band->corr[i*2];

        idx2 = band->corr[i*2+1];

        FFSWAP(uint8_t, band->rv_map->runtab[idx1], band->rv_map->runtab[idx2]);

        FFSWAP(int16_t, band->rv_map->valtab[idx1], band->rv_map->valtab[idx2]);

    }



    if (IVI_DEBUG && band->checksum_present) {

        chksum = ivi_calc_band_checksum(band);

        if (chksum != band->checksum) {

            av_log(avctx, AV_LOG_ERROR,

                   "Band checksum mismatch! Plane %d, band %d, received: %x, calculated: %x\n",

                   band->plane, band->band_num, band->checksum, chksum);

        }

    }



    return result;

}
