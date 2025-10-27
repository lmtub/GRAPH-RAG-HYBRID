static void filter_mb( H264Context *h, int mb_x, int mb_y, uint8_t *img_y, uint8_t *img_cb, uint8_t *img_cr) {

    MpegEncContext * const s = &h->s;

    const int mb_xy= mb_x + mb_y*s->mb_stride;

    int linesize, uvlinesize;

    int dir;



    /* FIXME Implement deblocking filter for field MB */

    if( h->sps.mb_aff ) {

        return;

    }

    linesize = s->linesize;

    uvlinesize = s->uvlinesize;



    /* dir : 0 -> vertical edge, 1 -> horizontal edge */

    for( dir = 0; dir < 2; dir++ )

    {

        int edge;

        const int mbm_xy = dir == 0 ? mb_xy -1 : mb_xy - s->mb_stride;

        int start = h->slice_table[mbm_xy] == 255 ? 1 : 0;



        if (h->deblocking_filter==2 && h->slice_table[mbm_xy] != h->slice_table[mb_xy])

            start = 1;



        /* Calculate bS */

        for( edge = start; edge < 4; edge++ ) {

            /* mbn_xy: neighbour macroblock (how that works for field ?) */

            int mbn_xy = edge > 0 ? mb_xy : mbm_xy;

            int bS[4];

            int qp;



            if( IS_INTRA( s->current_picture.mb_type[mb_xy] ) ||

                IS_INTRA( s->current_picture.mb_type[mbn_xy] ) ) {

                bS[0] = bS[1] = bS[2] = bS[3] = ( edge == 0 ? 4 : 3 );

            } else {

                int i;

                for( i = 0; i < 4; i++ ) {

                    int x = dir == 0 ? edge : i;

                    int y = dir == 0 ? i    : edge;

                    int b_idx= 8 + 4 + x + 8*y;

                    int bn_idx= b_idx - (dir ? 8:1);



                    if( h->non_zero_count_cache[b_idx] != 0 ||

                        h->non_zero_count_cache[bn_idx] != 0 ) {

                        bS[i] = 2;

                    }

                    else if( h->slice_type == P_TYPE ) {

                        if( h->ref_cache[0][b_idx] != h->ref_cache[0][bn_idx] ||

                            ABS( h->mv_cache[0][b_idx][0] - h->mv_cache[0][bn_idx][0] ) >= 4 ||

                            ABS( h->mv_cache[0][b_idx][1] - h->mv_cache[0][bn_idx][1] ) >= 4 )

                            bS[i] = 1;

                        else

                            bS[i] = 0;

                    } else {

                        /* FIXME Add support for B frame */

                        return;

                    }

                }



                if(bS[0]+bS[1]+bS[2]+bS[3] == 0)

                    continue;

            }



            /* Filter edge */

            // Do not use s->qscale as luma quantiser because it has not the same

            // value in IPCM macroblocks.

            qp = ( s->current_picture.qscale_table[mb_xy] + s->current_picture.qscale_table[mbn_xy] + 1 ) >> 1;

            //tprintf("filter mb:%d/%d dir:%d edge:%d, QPy:%d, QPc:%d, QPcn:%d\n", mb_x, mb_y, dir, edge, qp, h->chroma_qp, s->current_picture.qscale_table[mbn_xy]);

            if( dir == 0 ) {

                filter_mb_edgev( h, &img_y[4*edge], linesize, bS, qp );

                if( (edge&1) == 0 ) {

                    int chroma_qp = ( h->chroma_qp +

                                      get_chroma_qp( h, s->current_picture.qscale_table[mbn_xy] ) + 1 ) >> 1;

                    filter_mb_edgecv( h, &img_cb[2*edge], uvlinesize, bS, chroma_qp );

                    filter_mb_edgecv( h, &img_cr[2*edge], uvlinesize, bS, chroma_qp );

                }

            } else {

                filter_mb_edgeh( h, &img_y[4*edge*linesize], linesize, bS, qp );

                if( (edge&1) == 0 ) {

                    int chroma_qp = ( h->chroma_qp +

                                      get_chroma_qp( h, s->current_picture.qscale_table[mbn_xy] ) + 1 ) >> 1;

                    filter_mb_edgech( h, &img_cb[2*edge*uvlinesize], uvlinesize, bS, chroma_qp );

                    filter_mb_edgech( h, &img_cr[2*edge*uvlinesize], uvlinesize, bS, chroma_qp );

                }

            }

        }

    }

}
