void ff_h264_filter_mb(const H264Context *h, H264SliceContext *sl,

                       int mb_x, int mb_y,

                       uint8_t *img_y, uint8_t *img_cb, uint8_t *img_cr,

                       unsigned int linesize, unsigned int uvlinesize)

{

    const int mb_xy= mb_x + mb_y*h->mb_stride;

    const int mb_type = h->cur_pic.mb_type[mb_xy];

    const int mvy_limit = IS_INTERLACED(mb_type) ? 2 : 4;

    int first_vertical_edge_done = 0;

    int chroma = !(CONFIG_GRAY && (h->flags & AV_CODEC_FLAG_GRAY));

    int qp_bd_offset = 6 * (h->sps.bit_depth_luma - 8);

    int a = 52 + sl->slice_alpha_c0_offset - qp_bd_offset;

    int b = 52 + sl->slice_beta_offset - qp_bd_offset;



    if (FRAME_MBAFF(h)

            // and current and left pair do not have the same interlaced type

            && IS_INTERLACED(mb_type ^ sl->left_type[LTOP])

            // and left mb is in available to us

            && sl->left_type[LTOP]) {

        /* First vertical edge is different in MBAFF frames

         * There are 8 different bS to compute and 2 different Qp

         */

        DECLARE_ALIGNED(8, int16_t, bS)[8];

        int qp[2];

        int bqp[2];

        int rqp[2];

        int mb_qp, mbn0_qp, mbn1_qp;

        int i;

        first_vertical_edge_done = 1;



        if( IS_INTRA(mb_type) ) {

            AV_WN64A(&bS[0], 0x0004000400040004ULL);

            AV_WN64A(&bS[4], 0x0004000400040004ULL);

        } else {

            static const uint8_t offset[2][2][8]={

                {

                    {3+4*0, 3+4*0, 3+4*0, 3+4*0, 3+4*1, 3+4*1, 3+4*1, 3+4*1},

                    {3+4*2, 3+4*2, 3+4*2, 3+4*2, 3+4*3, 3+4*3, 3+4*3, 3+4*3},

                },{

                    {3+4*0, 3+4*1, 3+4*2, 3+4*3, 3+4*0, 3+4*1, 3+4*2, 3+4*3},

                    {3+4*0, 3+4*1, 3+4*2, 3+4*3, 3+4*0, 3+4*1, 3+4*2, 3+4*3},

                }

            };

            const uint8_t *off= offset[MB_FIELD(sl)][mb_y&1];

            for( i = 0; i < 8; i++ ) {

                int j= MB_FIELD(sl) ? i>>2 : i&1;

                int mbn_xy = sl->left_mb_xy[LEFT(j)];

                int mbn_type = sl->left_type[LEFT(j)];



                if( IS_INTRA( mbn_type ) )

                    bS[i] = 4;

                else{

                    bS[i] = 1 + !!(sl->non_zero_count_cache[12+8*(i>>1)] |

                         ((!h->pps.cabac && IS_8x8DCT(mbn_type)) ?

                            (h->cbp_table[mbn_xy] & (((MB_FIELD(sl) ? (i&2) : (mb_y&1)) ? 8 : 2) << 12))

                                                                       :

                            h->non_zero_count[mbn_xy][ off[i] ]));

                }

            }

        }



        mb_qp   = h->cur_pic.qscale_table[mb_xy];

        mbn0_qp = h->cur_pic.qscale_table[sl->left_mb_xy[0]];

        mbn1_qp = h->cur_pic.qscale_table[sl->left_mb_xy[1]];

        qp[0] = ( mb_qp + mbn0_qp + 1 ) >> 1;

        bqp[0] = ( get_chroma_qp( h, 0, mb_qp ) +

                   get_chroma_qp( h, 0, mbn0_qp ) + 1 ) >> 1;

        rqp[0] = ( get_chroma_qp( h, 1, mb_qp ) +

                   get_chroma_qp( h, 1, mbn0_qp ) + 1 ) >> 1;

        qp[1] = ( mb_qp + mbn1_qp + 1 ) >> 1;

        bqp[1] = ( get_chroma_qp( h, 0, mb_qp ) +

                   get_chroma_qp( h, 0, mbn1_qp ) + 1 ) >> 1;

        rqp[1] = ( get_chroma_qp( h, 1, mb_qp ) +

                   get_chroma_qp( h, 1, mbn1_qp ) + 1 ) >> 1;



        /* Filter edge */

        ff_tlog(h->avctx, "filter mb:%d/%d MBAFF, QPy:%d/%d, QPb:%d/%d QPr:%d/%d ls:%d uvls:%d", mb_x, mb_y, qp[0], qp[1], bqp[0], bqp[1], rqp[0], rqp[1], linesize, uvlinesize);

        { int i; for (i = 0; i < 8; i++) ff_tlog(h->avctx, " bS[%d]:%d", i, bS[i]); ff_tlog(h->avctx, "\n"); }

        if (MB_FIELD(sl)) {

            filter_mb_mbaff_edgev ( h, img_y                ,   linesize, bS  , 1, qp [0], a, b, 1 );

            filter_mb_mbaff_edgev ( h, img_y  + 8*  linesize,   linesize, bS+4, 1, qp [1], a, b, 1 );

            if (chroma){

                if (CHROMA444(h)) {

                    filter_mb_mbaff_edgev ( h, img_cb,                uvlinesize, bS  , 1, bqp[0], a, b, 1 );

                    filter_mb_mbaff_edgev ( h, img_cb + 8*uvlinesize, uvlinesize, bS+4, 1, bqp[1], a, b, 1 );

                    filter_mb_mbaff_edgev ( h, img_cr,                uvlinesize, bS  , 1, rqp[0], a, b, 1 );

                    filter_mb_mbaff_edgev ( h, img_cr + 8*uvlinesize, uvlinesize, bS+4, 1, rqp[1], a, b, 1 );

                } else if (CHROMA422(h)) {

                    filter_mb_mbaff_edgecv(h, img_cb,                uvlinesize, bS  , 1, bqp[0], a, b, 1);

                    filter_mb_mbaff_edgecv(h, img_cb + 8*uvlinesize, uvlinesize, bS+4, 1, bqp[1], a, b, 1);

                    filter_mb_mbaff_edgecv(h, img_cr,                uvlinesize, bS  , 1, rqp[0], a, b, 1);

                    filter_mb_mbaff_edgecv(h, img_cr + 8*uvlinesize, uvlinesize, bS+4, 1, rqp[1], a, b, 1);

                }else{

                    filter_mb_mbaff_edgecv( h, img_cb,                uvlinesize, bS  , 1, bqp[0], a, b, 1 );

                    filter_mb_mbaff_edgecv( h, img_cb + 4*uvlinesize, uvlinesize, bS+4, 1, bqp[1], a, b, 1 );

                    filter_mb_mbaff_edgecv( h, img_cr,                uvlinesize, bS  , 1, rqp[0], a, b, 1 );

                    filter_mb_mbaff_edgecv( h, img_cr + 4*uvlinesize, uvlinesize, bS+4, 1, rqp[1], a, b, 1 );

                }

            }

        }else{

            filter_mb_mbaff_edgev ( h, img_y              , 2*  linesize, bS  , 2, qp [0], a, b, 1 );

            filter_mb_mbaff_edgev ( h, img_y  +   linesize, 2*  linesize, bS+1, 2, qp [1], a, b, 1 );

            if (chroma){

                if (CHROMA444(h)) {

                    filter_mb_mbaff_edgev ( h, img_cb,              2*uvlinesize, bS  , 2, bqp[0], a, b, 1 );

                    filter_mb_mbaff_edgev ( h, img_cb + uvlinesize, 2*uvlinesize, bS+1, 2, bqp[1], a, b, 1 );

                    filter_mb_mbaff_edgev ( h, img_cr,              2*uvlinesize, bS  , 2, rqp[0], a, b, 1 );

                    filter_mb_mbaff_edgev ( h, img_cr + uvlinesize, 2*uvlinesize, bS+1, 2, rqp[1], a, b, 1 );

                }else{

                    filter_mb_mbaff_edgecv( h, img_cb,              2*uvlinesize, bS  , 2, bqp[0], a, b, 1 );

                    filter_mb_mbaff_edgecv( h, img_cb + uvlinesize, 2*uvlinesize, bS+1, 2, bqp[1], a, b, 1 );

                    filter_mb_mbaff_edgecv( h, img_cr,              2*uvlinesize, bS  , 2, rqp[0], a, b, 1 );

                    filter_mb_mbaff_edgecv( h, img_cr + uvlinesize, 2*uvlinesize, bS+1, 2, rqp[1], a, b, 1 );

                }

            }

        }

    }



#if CONFIG_SMALL

    {

        int dir;

        for (dir = 0; dir < 2; dir++)

            filter_mb_dir(h, sl, mb_x, mb_y, img_y, img_cb, img_cr, linesize,

                          uvlinesize, mb_xy, mb_type, mvy_limit,

                          dir ? 0 : first_vertical_edge_done, a, b,

                          chroma, dir);

    }

#else

    filter_mb_dir(h, sl, mb_x, mb_y, img_y, img_cb, img_cr, linesize, uvlinesize, mb_xy, mb_type, mvy_limit, first_vertical_edge_done, a, b, chroma, 0);

    filter_mb_dir(h, sl, mb_x, mb_y, img_y, img_cb, img_cr, linesize, uvlinesize, mb_xy, mb_type, mvy_limit, 0,                        a, b, chroma, 1);

#endif

}
