static int decode_gop_header(IVI5DecContext *ctx, AVCodecContext *avctx)

{

    int             result, i, p, tile_size, pic_size_indx, mb_size, blk_size;

    int             quant_mat, blk_size_changed = 0;

    IVIBandDesc     *band, *band1, *band2;

    IVIPicConfig    pic_conf;



    ctx->gop_flags = get_bits(&ctx->gb, 8);



    ctx->gop_hdr_size = (ctx->gop_flags & 1) ? get_bits(&ctx->gb, 16) : 0;



    if (ctx->gop_flags & IVI5_IS_PROTECTED)

        ctx->lock_word = get_bits_long(&ctx->gb, 32);



    tile_size = (ctx->gop_flags & 0x40) ? 64 << get_bits(&ctx->gb, 2) : 0;

    if (tile_size > 256) {

        av_log(avctx, AV_LOG_ERROR, "Invalid tile size: %d\n", tile_size);

        return -1;

    }



    /* decode number of wavelet bands */

    /* num_levels * 3 + 1 */

    pic_conf.luma_bands   = get_bits(&ctx->gb, 2) * 3 + 1;

    pic_conf.chroma_bands = get_bits1(&ctx->gb)   * 3 + 1;

    ctx->is_scalable = pic_conf.luma_bands != 1 || pic_conf.chroma_bands != 1;

    if (ctx->is_scalable && (pic_conf.luma_bands != 4 || pic_conf.chroma_bands != 1)) {

        av_log(avctx, AV_LOG_ERROR, "Scalability: unsupported subdivision! Luma bands: %d, chroma bands: %d\n",

               pic_conf.luma_bands, pic_conf.chroma_bands);

        return -1;

    }



    pic_size_indx = get_bits(&ctx->gb, 4);

    if (pic_size_indx == IVI5_PIC_SIZE_ESC) {

        pic_conf.pic_height = get_bits(&ctx->gb, 13);

        pic_conf.pic_width  = get_bits(&ctx->gb, 13);

    } else {

        pic_conf.pic_height = ivi5_common_pic_sizes[pic_size_indx * 2 + 1] << 2;

        pic_conf.pic_width  = ivi5_common_pic_sizes[pic_size_indx * 2    ] << 2;

    }



    if (ctx->gop_flags & 2) {

        av_log(avctx, AV_LOG_ERROR, "YV12 picture format not supported!\n");

        return -1;

    }



    pic_conf.chroma_height = (pic_conf.pic_height + 3) >> 2;

    pic_conf.chroma_width  = (pic_conf.pic_width  + 3) >> 2;



    if (!tile_size) {

        pic_conf.tile_height = pic_conf.pic_height;

        pic_conf.tile_width  = pic_conf.pic_width;

    } else {

        pic_conf.tile_height = pic_conf.tile_width = tile_size;

    }



    /* check if picture layout was changed and reallocate buffers */

    if (ivi_pic_config_cmp(&pic_conf, &ctx->pic_conf)) {

        result = ff_ivi_init_planes(ctx->planes, &pic_conf);

        if (result) {

            av_log(avctx, AV_LOG_ERROR, "Couldn't reallocate color planes!\n");

            return -1;

        }

        ctx->pic_conf = pic_conf;

        blk_size_changed = 1; /* force reallocation of the internal structures */

    }



    for (p = 0; p <= 1; p++) {

        for (i = 0; i < (!p ? pic_conf.luma_bands : pic_conf.chroma_bands); i++) {

            band = &ctx->planes[p].bands[i];



            band->is_halfpel = get_bits1(&ctx->gb);



            mb_size  = get_bits1(&ctx->gb);

            blk_size = 8 >> get_bits1(&ctx->gb);

            mb_size  = blk_size << !mb_size;



            blk_size_changed = mb_size != band->mb_size || blk_size != band->blk_size;

            if (blk_size_changed) {

                band->mb_size  = mb_size;

                band->blk_size = blk_size;

            }



            if (get_bits1(&ctx->gb)) {

                av_log(avctx, AV_LOG_ERROR, "Extended transform info encountered!\n");

                return -1;

            }



            /* select transform function and scan pattern according to plane and band number */

            switch ((p << 2) + i) {

            case 0:

                band->inv_transform = ff_ivi_inverse_slant_8x8;

                band->dc_transform  = ff_ivi_dc_slant_2d;

                band->scan          = ff_zigzag_direct;

                break;



            case 1:

                band->inv_transform = ff_ivi_row_slant8;

                band->dc_transform  = ff_ivi_dc_row_slant;

                band->scan          = ff_ivi_vertical_scan_8x8;

                break;



            case 2:

                band->inv_transform = ff_ivi_col_slant8;

                band->dc_transform  = ff_ivi_dc_col_slant;

                band->scan          = ff_ivi_horizontal_scan_8x8;

                break;



            case 3:

                band->inv_transform = ff_ivi_put_pixels_8x8;

                band->dc_transform  = ff_ivi_put_dc_pixel_8x8;

                band->scan          = ff_ivi_horizontal_scan_8x8;

                break;



            case 4:

                band->inv_transform = ff_ivi_inverse_slant_4x4;

                band->dc_transform  = ff_ivi_dc_slant_2d;

                band->scan          = ff_ivi_direct_scan_4x4;

                break;

            }



            band->is_2d_trans = band->inv_transform == ff_ivi_inverse_slant_8x8 ||

                                band->inv_transform == ff_ivi_inverse_slant_4x4;



            /* select dequant matrix according to plane and band number */

            if (!p) {

                quant_mat = (pic_conf.luma_bands > 1) ? i+1 : 0;

            } else {

                quant_mat = 5;

            }



            if (band->blk_size == 8) {

                band->intra_base  = &ivi5_base_quant_8x8_intra[quant_mat][0];

                band->inter_base  = &ivi5_base_quant_8x8_inter[quant_mat][0];

                band->intra_scale = &ivi5_scale_quant_8x8_intra[quant_mat][0];

                band->inter_scale = &ivi5_scale_quant_8x8_inter[quant_mat][0];

            } else {

                band->intra_base  = ivi5_base_quant_4x4_intra;

                band->inter_base  = ivi5_base_quant_4x4_inter;

                band->intra_scale = ivi5_scale_quant_4x4_intra;

                band->inter_scale = ivi5_scale_quant_4x4_inter;

            }



            if (get_bits(&ctx->gb, 2)) {

                av_log(avctx, AV_LOG_ERROR, "End marker missing!\n");

                return -1;

            }

        }

    }



    /* copy chroma parameters into the 2nd chroma plane */

    for (i = 0; i < pic_conf.chroma_bands; i++) {

        band1 = &ctx->planes[1].bands[i];

        band2 = &ctx->planes[2].bands[i];



        band2->width         = band1->width;

        band2->height        = band1->height;

        band2->mb_size       = band1->mb_size;

        band2->blk_size      = band1->blk_size;

        band2->is_halfpel    = band1->is_halfpel;

        band2->intra_base    = band1->intra_base;

        band2->inter_base    = band1->inter_base;

        band2->intra_scale   = band1->intra_scale;

        band2->inter_scale   = band1->inter_scale;

        band2->scan          = band1->scan;

        band2->inv_transform = band1->inv_transform;

        band2->dc_transform  = band1->dc_transform;

        band2->is_2d_trans   = band1->is_2d_trans;

    }



    /* reallocate internal structures if needed */

    if (blk_size_changed) {

        result = ff_ivi_init_tiles(ctx->planes, pic_conf.tile_width,

                                   pic_conf.tile_height);

        if (result) {

            av_log(avctx, AV_LOG_ERROR,

                   "Couldn't reallocate internal structures!\n");

            return -1;

        }

    }



    if (ctx->gop_flags & 8) {

        if (get_bits(&ctx->gb, 3)) {

            av_log(avctx, AV_LOG_ERROR, "Alignment bits are not zero!\n");

            return -1;

        }



        if (get_bits1(&ctx->gb))

            skip_bits_long(&ctx->gb, 24); /* skip transparency fill color */

    }



    align_get_bits(&ctx->gb);



    skip_bits(&ctx->gb, 23); /* FIXME: unknown meaning */



    /* skip GOP extension if any */

    if (get_bits1(&ctx->gb)) {

        do {

            i = get_bits(&ctx->gb, 16);

        } while (i & 0x8000);

    }



    align_get_bits(&ctx->gb);



    return 0;

}
