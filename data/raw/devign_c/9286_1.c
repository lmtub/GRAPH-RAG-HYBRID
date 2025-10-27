static int decode_band(IVI45DecContext *ctx,

                       IVIBandDesc *band, AVCodecContext *avctx)

{

    int         result, i, t, idx1, idx2, pos;

    IVITile     *tile;



    band->buf     = band->bufs[ctx->dst_buf];

    if (!band->buf) {

        av_log(avctx, AV_LOG_ERROR, "Band buffer points to no data!\n");

        return AVERROR_INVALIDDATA;

    }

    band->ref_buf = band->bufs[ctx->ref_buf];

    band->data_ptr = ctx->frame_data + (get_bits_count(&ctx->gb) >> 3);



    result = ctx->decode_band_hdr(ctx, band, avctx);

    if (result) {

        av_log(avctx, AV_LOG_ERROR, "Error while decoding band header: %d\n",

               result);

        return result;

    }



    if (band->is_empty) {

        av_log(avctx, AV_LOG_ERROR, "Empty band encountered!\n");

        return AVERROR_INVALIDDATA;

    }



    band->rv_map = &ctx->rvmap_tabs[band->rvmap_sel];



    /* apply corrections to the selected rvmap table if present */

    for (i = 0; i < band->num_corr; i++) {

        idx1 = band->corr[i * 2];

        idx2 = band->corr[i * 2 + 1];

        FFSWAP(uint8_t, band->rv_map->runtab[idx1], band->rv_map->runtab[idx2]);

        FFSWAP(int16_t, band->rv_map->valtab[idx1], band->rv_map->valtab[idx2]);

    }



    pos = get_bits_count(&ctx->gb);



    for (t = 0; t < band->num_tiles; t++) {

        tile = &band->tiles[t];



        if (tile->mb_size != band->mb_size) {

            av_log(avctx, AV_LOG_ERROR, "MB sizes mismatch: %d vs. %d\n",

                   band->mb_size, tile->mb_size);

            return AVERROR_INVALIDDATA;

        }

        tile->is_empty = get_bits1(&ctx->gb);

        if (tile->is_empty) {

            result = ivi_process_empty_tile(avctx, band, tile,

                                      (ctx->planes[0].bands[0].mb_size >> 3) - (band->mb_size >> 3));

            if (result < 0)

                break;

            av_dlog(avctx, "Empty tile encountered!\n");

        } else {

            tile->data_size = ivi_dec_tile_data_size(&ctx->gb);

            if (!tile->data_size) {

                av_log(avctx, AV_LOG_ERROR, "Tile data size is zero!\n");

                return AVERROR_INVALIDDATA;

            }



            result = ctx->decode_mb_info(ctx, band, tile, avctx);

            if (result < 0)

                break;



            result = ivi_decode_blocks(&ctx->gb, band, tile, avctx);

            if (result < 0 || ((get_bits_count(&ctx->gb) - pos) >> 3) != tile->data_size) {

                av_log(avctx, AV_LOG_ERROR, "Corrupted tile data encountered!\n");

                break;

            }



            pos += tile->data_size << 3; // skip to next tile

        }

    }



    /* restore the selected rvmap table by applying its corrections in reverse order */

    for (i = band->num_corr-1; i >= 0; i--) {

        idx1 = band->corr[i*2];

        idx2 = band->corr[i*2+1];

        FFSWAP(uint8_t, band->rv_map->runtab[idx1], band->rv_map->runtab[idx2]);

        FFSWAP(int16_t, band->rv_map->valtab[idx1], band->rv_map->valtab[idx2]);

    }



#ifdef DEBUG

    if (band->checksum_present) {

        uint16_t chksum = ivi_calc_band_checksum(band);

        if (chksum != band->checksum) {

            av_log(avctx, AV_LOG_ERROR,

                   "Band checksum mismatch! Plane %d, band %d, received: %x, calculated: %x\n",

                   band->plane, band->band_num, band->checksum, chksum);

        }

    }

#endif



    align_get_bits(&ctx->gb);



    return result;

}
