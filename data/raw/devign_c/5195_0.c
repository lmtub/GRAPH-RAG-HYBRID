static void av_always_inline filter_mb_dir(H264Context *h, int mb_x, int mb_y, uint8_t *img_y, uint8_t *img_cb, uint8_t *img_cr, unsigned int linesize, unsigned int uvlinesize, int mb_xy, int mb_type, int mvy_limit, int first_vertical_edge_done, int dir) {

    MpegEncContext * const s = &h->s;

    int edge;

    const int mbm_xy = dir == 0 ? mb_xy -1 : h->top_mb_xy;

    const int mbm_type = s->current_picture.mb_type[mbm_xy];

    int (*ref2frm) [64] = h->ref2frm[ h->slice_num          &(MAX_SLICES-1) ][0] + (MB_MBAFF ? 20 : 2);

    int (*ref2frmm)[64] = h->ref2frm[ h->slice_table[mbm_xy]&(MAX_SLICES-1) ][0] + (MB_MBAFF ? 20 : 2);

    int start = h->slice_table[mbm_xy] == 0xFFFF ? 1 : 0;



    const int edges = (mb_type & (MB_TYPE_16x16|MB_TYPE_SKIP))

                              == (MB_TYPE_16x16|MB_TYPE_SKIP) ? 1 : 4;

    // how often to recheck mv-based bS when iterating between edges

    const int mask_edge = (mb_type & (MB_TYPE_16x16 | (MB_TYPE_16x8 << dir))) ? 3 :

                          (mb_type & (MB_TYPE_8x16 >> dir)) ? 1 : 0;

    // how often to recheck mv-based bS when iterating along each edge

    const int mask_par0 = mb_type & (MB_TYPE_16x16 | (MB_TYPE_8x16 >> dir));



    if (first_vertical_edge_done) {

        start = 1;

    }



    if (h->deblocking_filter==2 && h->slice_table[mbm_xy] != h->slice_table[mb_xy])

        start = 1;



    if (FRAME_MBAFF && (dir == 1) && ((mb_y&1) == 0) && start == 0

        && !IS_INTERLACED(mb_type)

        && IS_INTERLACED(mbm_type)

        ) {

        // This is a special case in the norm where the filtering must

        // be done twice (one each of the field) even if we are in a

        // frame macroblock.

        //

        static const int nnz_idx[4] = {4,5,6,3};

        unsigned int tmp_linesize   = 2 *   linesize;

        unsigned int tmp_uvlinesize = 2 * uvlinesize;

        int mbn_xy = mb_xy - 2 * s->mb_stride;

        int qp;

        int i, j;

        int16_t bS[4];



        for(j=0; j<2; j++, mbn_xy += s->mb_stride){

            if( IS_INTRA(mb_type) ||

                IS_INTRA(s->current_picture.mb_type[mbn_xy]) ) {

                bS[0] = bS[1] = bS[2] = bS[3] = 3;

            } else {

                const uint8_t *mbn_nnz = h->non_zero_count[mbn_xy];

                for( i = 0; i < 4; i++ ) {

                    if( h->non_zero_count_cache[scan8[0]+i] != 0 ||

                        mbn_nnz[nnz_idx[i]] != 0 )

                        bS[i] = 2;

                    else

                        bS[i] = 1;

                }

            }

            // Do not use s->qscale as luma quantizer because it has not the same

            // value in IPCM macroblocks.

            qp = ( s->current_picture.qscale_table[mb_xy] + s->current_picture.qscale_table[mbn_xy] + 1 ) >> 1;

            tprintf(s->avctx, "filter mb:%d/%d dir:%d edge:%d, QPy:%d ls:%d uvls:%d", mb_x, mb_y, dir, edge, qp, tmp_linesize, tmp_uvlinesize);

            { int i; for (i = 0; i < 4; i++) tprintf(s->avctx, " bS[%d]:%d", i, bS[i]); tprintf(s->avctx, "\n"); }

            filter_mb_edgeh( h, &img_y[j*linesize], tmp_linesize, bS, qp );

            filter_mb_edgech( h, &img_cb[j*uvlinesize], tmp_uvlinesize, bS,

                              ( h->chroma_qp[0] + get_chroma_qp( h, 0, s->current_picture.qscale_table[mbn_xy] ) + 1 ) >> 1);

            filter_mb_edgech( h, &img_cr[j*uvlinesize], tmp_uvlinesize, bS,

                              ( h->chroma_qp[1] + get_chroma_qp( h, 1, s->current_picture.qscale_table[mbn_xy] ) + 1 ) >> 1);

        }



        start = 1;

    }



    /* Calculate bS */

    for( edge = start; edge < edges; edge++ ) {

        /* mbn_xy: neighbor macroblock */

        const int mbn_xy = edge > 0 ? mb_xy : mbm_xy;

        const int mbn_type = s->current_picture.mb_type[mbn_xy];

        int (*ref2frmn)[64] = edge > 0 ? ref2frm : ref2frmm;

        int16_t bS[4];

        int qp;



        if( (edge&1) && IS_8x8DCT(mb_type) )

            continue;



        if( IS_INTRA(mb_type) ||

            IS_INTRA(mbn_type) ) {

            int value;

            if (edge == 0) {

                if (   (!IS_INTERLACED(mb_type) && !IS_INTERLACED(mbm_type))

                    || ((FRAME_MBAFF || (s->picture_structure != PICT_FRAME)) && (dir == 0))

                ) {

                    value = 4;

                } else {

                    value = 3;

                }

            } else {

                value = 3;

            }

            bS[0] = bS[1] = bS[2] = bS[3] = value;

        } else {

            int i, l;

            int mv_done;



            if( edge & mask_edge ) {

                bS[0] = bS[1] = bS[2] = bS[3] = 0;

                mv_done = 1;

            }

            else if( FRAME_MBAFF && IS_INTERLACED(mb_type ^ mbn_type)) {

                bS[0] = bS[1] = bS[2] = bS[3] = 1;

                mv_done = 1;

            }

            else if( mask_par0 && (edge || (mbn_type & (MB_TYPE_16x16 | (MB_TYPE_8x16 >> dir)))) ) {

                int b_idx= 8 + 4 + edge * (dir ? 8:1);

                int bn_idx= b_idx - (dir ? 8:1);

                int v = 0;



                for( l = 0; !v && l < 1 + (h->slice_type_nos == FF_B_TYPE); l++ ) {

                    v |= ref2frm[l][h->ref_cache[l][b_idx]] != ref2frmn[l][h->ref_cache[l][bn_idx]] ||

                         FFABS( h->mv_cache[l][b_idx][0] - h->mv_cache[l][bn_idx][0] ) >= 4 ||

                         FFABS( h->mv_cache[l][b_idx][1] - h->mv_cache[l][bn_idx][1] ) >= mvy_limit;

                }



                if(h->slice_type_nos == FF_B_TYPE && v){

                    v=0;

                    for( l = 0; !v && l < 2; l++ ) {

                        int ln= 1-l;

                        v |= ref2frm[l][h->ref_cache[l][b_idx]] != ref2frmn[ln][h->ref_cache[ln][bn_idx]] ||

                            FFABS( h->mv_cache[l][b_idx][0] - h->mv_cache[ln][bn_idx][0] ) >= 4 ||

                            FFABS( h->mv_cache[l][b_idx][1] - h->mv_cache[ln][bn_idx][1] ) >= mvy_limit;

                    }

                }



                bS[0] = bS[1] = bS[2] = bS[3] = v;

                mv_done = 1;

            }

            else

                mv_done = 0;



            for( i = 0; i < 4; i++ ) {

                int x = dir == 0 ? edge : i;

                int y = dir == 0 ? i    : edge;

                int b_idx= 8 + 4 + x + 8*y;

                int bn_idx= b_idx - (dir ? 8:1);



                if( h->non_zero_count_cache[b_idx] != 0 ||

                    h->non_zero_count_cache[bn_idx] != 0 ) {

                    bS[i] = 2;

                }

                else if(!mv_done)

                {

                    bS[i] = 0;

                    for( l = 0; l < 1 + (h->slice_type_nos == FF_B_TYPE); l++ ) {

                        if( ref2frm[l][h->ref_cache[l][b_idx]] != ref2frmn[l][h->ref_cache[l][bn_idx]] ||

                            FFABS( h->mv_cache[l][b_idx][0] - h->mv_cache[l][bn_idx][0] ) >= 4 ||

                            FFABS( h->mv_cache[l][b_idx][1] - h->mv_cache[l][bn_idx][1] ) >= mvy_limit ) {

                            bS[i] = 1;

                            break;

                        }

                    }



                    if(h->slice_type_nos == FF_B_TYPE && bS[i]){

                        bS[i] = 0;

                        for( l = 0; l < 2; l++ ) {

                            int ln= 1-l;

                            if( ref2frm[l][h->ref_cache[l][b_idx]] != ref2frmn[ln][h->ref_cache[ln][bn_idx]] ||

                                FFABS( h->mv_cache[l][b_idx][0] - h->mv_cache[ln][bn_idx][0] ) >= 4 ||

                                FFABS( h->mv_cache[l][b_idx][1] - h->mv_cache[ln][bn_idx][1] ) >= mvy_limit ) {

                                bS[i] = 1;

                                break;

                            }

                        }

                    }

                }

            }



            if(bS[0]+bS[1]+bS[2]+bS[3] == 0)

                continue;

        }



        /* Filter edge */

        // Do not use s->qscale as luma quantizer because it has not the same

        // value in IPCM macroblocks.

        qp = ( s->current_picture.qscale_table[mb_xy] + s->current_picture.qscale_table[mbn_xy] + 1 ) >> 1;

        //tprintf(s->avctx, "filter mb:%d/%d dir:%d edge:%d, QPy:%d, QPc:%d, QPcn:%d\n", mb_x, mb_y, dir, edge, qp, h->chroma_qp, s->current_picture.qscale_table[mbn_xy]);

        tprintf(s->avctx, "filter mb:%d/%d dir:%d edge:%d, QPy:%d ls:%d uvls:%d", mb_x, mb_y, dir, edge, qp, linesize, uvlinesize);

        { int i; for (i = 0; i < 4; i++) tprintf(s->avctx, " bS[%d]:%d", i, bS[i]); tprintf(s->avctx, "\n"); }

        if( dir == 0 ) {

            filter_mb_edgev( h, &img_y[4*edge], linesize, bS, qp );

            if( (edge&1) == 0 ) {

                filter_mb_edgecv( h, &img_cb[2*edge], uvlinesize, bS,

                                  ( h->chroma_qp[0] + get_chroma_qp( h, 0, s->current_picture.qscale_table[mbn_xy] ) + 1 ) >> 1);

                filter_mb_edgecv( h, &img_cr[2*edge], uvlinesize, bS,

                                  ( h->chroma_qp[1] + get_chroma_qp( h, 1, s->current_picture.qscale_table[mbn_xy] ) + 1 ) >> 1);

            }

        } else {

            filter_mb_edgeh( h, &img_y[4*edge*linesize], linesize, bS, qp );

            if( (edge&1) == 0 ) {

                filter_mb_edgech( h, &img_cb[2*edge*uvlinesize], uvlinesize, bS,

                                  ( h->chroma_qp[0] + get_chroma_qp( h, 0, s->current_picture.qscale_table[mbn_xy] ) + 1 ) >> 1);

                filter_mb_edgech( h, &img_cr[2*edge*uvlinesize], uvlinesize, bS,

                                  ( h->chroma_qp[1] + get_chroma_qp( h, 1, s->current_picture.qscale_table[mbn_xy] ) + 1 ) >> 1);

            }

        }

    }

}
