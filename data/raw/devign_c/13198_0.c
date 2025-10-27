av_cold int ff_ivi_init_tiles(IVIPlaneDesc *planes, int tile_width, int tile_height)

{

    int         p, b, x, y, x_tiles, y_tiles, t_width, t_height;

    IVIBandDesc *band;

    IVITile     *tile, *ref_tile;



    for (p = 0; p < 3; p++) {

        t_width  = !p ? tile_width  : (tile_width  + 3) >> 2;

        t_height = !p ? tile_height : (tile_height + 3) >> 2;



        if (!p && planes[0].num_bands == 4) {

            t_width  >>= 1;

            t_height >>= 1;

        }

        if(t_width<=0 || t_height<=0)

            return AVERROR(EINVAL);



        for (b = 0; b < planes[p].num_bands; b++) {

            band = &planes[p].bands[b];

            x_tiles = IVI_NUM_TILES(band->width, t_width);

            y_tiles = IVI_NUM_TILES(band->height, t_height);

            band->num_tiles = x_tiles * y_tiles;



            av_freep(&band->tiles);

            band->tiles = av_mallocz(band->num_tiles * sizeof(IVITile));

            if (!band->tiles)

                return AVERROR(ENOMEM);



            tile = band->tiles;



            /* use the first luma band as reference for motion vectors

             * and quant */

            ref_tile = planes[0].bands[0].tiles;



            for (y = 0; y < band->height; y += t_height) {

                for (x = 0; x < band->width; x += t_width) {

                    tile->xpos     = x;

                    tile->ypos     = y;

                    tile->mb_size  = band->mb_size;

                    tile->width    = FFMIN(band->width - x,  t_width);

                    tile->height   = FFMIN(band->height - y, t_height);

                    tile->is_empty = tile->data_size = 0;

                    /* calculate number of macroblocks */

                    tile->num_MBs  = IVI_MBs_PER_TILE(tile->width, tile->height,

                                                      band->mb_size);



                    av_freep(&tile->mbs);

                    tile->mbs = av_malloc(tile->num_MBs * sizeof(IVIMbInfo));

                    if (!tile->mbs)

                        return AVERROR(ENOMEM);



                    tile->ref_mbs = 0;

                    if (p || b) {

                        tile->ref_mbs = ref_tile->mbs;

                        ref_tile++;

                    }



                    tile++;

                }

            }



        }// for b

    }// for p



    return 0;

}
