static av_always_inline void h264_filter_mb_fast_internal(H264Context *h,

                                                          int mb_x, int mb_y,

                                                          uint8_t *img_y,

                                                          uint8_t *img_cb,

                                                          uint8_t *img_cr,

                                                          unsigned int linesize,

                                                          unsigned int uvlinesize,

                                                          int pixel_shift)

{

    int chroma = !(CONFIG_GRAY && (h->flags&CODEC_FLAG_GRAY));

    int chroma444 = CHROMA444(h);

    int chroma422 = CHROMA422(h);



    int mb_xy = h->mb_xy;

    int left_type= h->left_type[LTOP];

    int top_type= h->top_type;



    int qp_bd_offset = 6 * (h->sps.bit_depth_luma - 8);

    int a = h->slice_alpha_c0_offset - qp_bd_offset;

    int b = h->slice_beta_offset - qp_bd_offset;



    int mb_type = h->cur_pic.mb_type[mb_xy];

    int qp      = h->cur_pic.qscale_table[mb_xy];

    int qp0     = h->cur_pic.qscale_table[mb_xy - 1];

    int qp1     = h->cur_pic.qscale_table[h->top_mb_xy];

    int qpc = get_chroma_qp( h, 0, qp );

    int qpc0 = get_chroma_qp( h, 0, qp0 );

    int qpc1 = get_chroma_qp( h, 0, qp1 );

    qp0 = (qp + qp0 + 1) >> 1;

    qp1 = (qp + qp1 + 1) >> 1;

    qpc0 = (qpc + qpc0 + 1) >> 1;

    qpc1 = (qpc + qpc1 + 1) >> 1;



    if( IS_INTRA(mb_type) ) {

        static const int16_t bS4[4] = {4,4,4,4};

        static const int16_t bS3[4] = {3,3,3,3};

        const int16_t *bSH = FIELD_PICTURE(h) ? bS3 : bS4;

        if(left_type)

            filter_mb_edgev( &img_y[4*0<<pixel_shift], linesize, bS4, qp0, a, b, h, 1);

        if( IS_8x8DCT(mb_type) ) {

            filter_mb_edgev( &img_y[4*2<<pixel_shift], linesize, bS3, qp, a, b, h, 0);

            if(top_type){

                filter_mb_edgeh( &img_y[4*0*linesize], linesize, bSH, qp1, a, b, h, 1);

            }

            filter_mb_edgeh( &img_y[4*2*linesize], linesize, bS3, qp, a, b, h, 0);

        } else {

            filter_mb_edgev( &img_y[4*1<<pixel_shift], linesize, bS3, qp, a, b, h, 0);

            filter_mb_edgev( &img_y[4*2<<pixel_shift], linesize, bS3, qp, a, b, h, 0);

            filter_mb_edgev( &img_y[4*3<<pixel_shift], linesize, bS3, qp, a, b, h, 0);

            if(top_type){

                filter_mb_edgeh( &img_y[4*0*linesize], linesize, bSH, qp1, a, b, h, 1);

            }

            filter_mb_edgeh( &img_y[4*1*linesize], linesize, bS3, qp, a, b, h, 0);

            filter_mb_edgeh( &img_y[4*2*linesize], linesize, bS3, qp, a, b, h, 0);

            filter_mb_edgeh( &img_y[4*3*linesize], linesize, bS3, qp, a, b, h, 0);

        }

        if(chroma){

            if(chroma444){

                if(left_type){

                    filter_mb_edgev( &img_cb[4*0<<pixel_shift], linesize, bS4, qpc0, a, b, h, 1);

                    filter_mb_edgev( &img_cr[4*0<<pixel_shift], linesize, bS4, qpc0, a, b, h, 1);

                }

                if( IS_8x8DCT(mb_type) ) {

                    filter_mb_edgev( &img_cb[4*2<<pixel_shift], linesize, bS3, qpc, a, b, h, 0);

                    filter_mb_edgev( &img_cr[4*2<<pixel_shift], linesize, bS3, qpc, a, b, h, 0);

                    if(top_type){

                        filter_mb_edgeh( &img_cb[4*0*linesize], linesize, bSH, qpc1, a, b, h, 1 );

                        filter_mb_edgeh( &img_cr[4*0*linesize], linesize, bSH, qpc1, a, b, h, 1 );

                    }

                    filter_mb_edgeh( &img_cb[4*2*linesize], linesize, bS3, qpc, a, b, h, 0);

                    filter_mb_edgeh( &img_cr[4*2*linesize], linesize, bS3, qpc, a, b, h, 0);

                } else {

                    filter_mb_edgev( &img_cb[4*1<<pixel_shift], linesize, bS3, qpc, a, b, h, 0);

                    filter_mb_edgev( &img_cr[4*1<<pixel_shift], linesize, bS3, qpc, a, b, h, 0);

                    filter_mb_edgev( &img_cb[4*2<<pixel_shift], linesize, bS3, qpc, a, b, h, 0);

                    filter_mb_edgev( &img_cr[4*2<<pixel_shift], linesize, bS3, qpc, a, b, h, 0);

                    filter_mb_edgev( &img_cb[4*3<<pixel_shift], linesize, bS3, qpc, a, b, h, 0);

                    filter_mb_edgev( &img_cr[4*3<<pixel_shift], linesize, bS3, qpc, a, b, h, 0);

                    if(top_type){

                        filter_mb_edgeh( &img_cb[4*0*linesize], linesize, bSH, qpc1, a, b, h, 1);

                        filter_mb_edgeh( &img_cr[4*0*linesize], linesize, bSH, qpc1, a, b, h, 1);

                    }

                    filter_mb_edgeh( &img_cb[4*1*linesize], linesize, bS3, qpc, a, b, h, 0);

                    filter_mb_edgeh( &img_cr[4*1*linesize], linesize, bS3, qpc, a, b, h, 0);

                    filter_mb_edgeh( &img_cb[4*2*linesize], linesize, bS3, qpc, a, b, h, 0);

                    filter_mb_edgeh( &img_cr[4*2*linesize], linesize, bS3, qpc, a, b, h, 0);

                    filter_mb_edgeh( &img_cb[4*3*linesize], linesize, bS3, qpc, a, b, h, 0);

                    filter_mb_edgeh( &img_cr[4*3*linesize], linesize, bS3, qpc, a, b, h, 0);

                }

            }else if(chroma422){

                if(left_type){

                    filter_mb_edgecv(&img_cb[2*0<<pixel_shift], uvlinesize, bS4, qpc0, a, b, h, 1);

                    filter_mb_edgecv(&img_cr[2*0<<pixel_shift], uvlinesize, bS4, qpc0, a, b, h, 1);

                }

                filter_mb_edgecv(&img_cb[2*2<<pixel_shift], uvlinesize, bS3, qpc, a, b, h, 0);

                filter_mb_edgecv(&img_cr[2*2<<pixel_shift], uvlinesize, bS3, qpc, a, b, h, 0);

                if(top_type){

                    filter_mb_edgech(&img_cb[4*0*uvlinesize], uvlinesize, bSH, qpc1, a, b, h, 1);

                    filter_mb_edgech(&img_cr[4*0*uvlinesize], uvlinesize, bSH, qpc1, a, b, h, 1);

                }

                filter_mb_edgech(&img_cb[4*1*uvlinesize], uvlinesize, bS3, qpc, a, b, h, 0);

                filter_mb_edgech(&img_cr[4*1*uvlinesize], uvlinesize, bS3, qpc, a, b, h, 0);

                filter_mb_edgech(&img_cb[4*2*uvlinesize], uvlinesize, bS3, qpc, a, b, h, 0);

                filter_mb_edgech(&img_cr[4*2*uvlinesize], uvlinesize, bS3, qpc, a, b, h, 0);

                filter_mb_edgech(&img_cb[4*3*uvlinesize], uvlinesize, bS3, qpc, a, b, h, 0);

                filter_mb_edgech(&img_cr[4*3*uvlinesize], uvlinesize, bS3, qpc, a, b, h, 0);

            }else{

                if(left_type){

                    filter_mb_edgecv( &img_cb[2*0<<pixel_shift], uvlinesize, bS4, qpc0, a, b, h, 1);

                    filter_mb_edgecv( &img_cr[2*0<<pixel_shift], uvlinesize, bS4, qpc0, a, b, h, 1);

                }

                filter_mb_edgecv( &img_cb[2*2<<pixel_shift], uvlinesize, bS3, qpc, a, b, h, 0);

                filter_mb_edgecv( &img_cr[2*2<<pixel_shift], uvlinesize, bS3, qpc, a, b, h, 0);

                if(top_type){

                    filter_mb_edgech( &img_cb[2*0*uvlinesize], uvlinesize, bSH, qpc1, a, b, h, 1);

                    filter_mb_edgech( &img_cr[2*0*uvlinesize], uvlinesize, bSH, qpc1, a, b, h, 1);

                }

                filter_mb_edgech( &img_cb[2*2*uvlinesize], uvlinesize, bS3, qpc, a, b, h, 0);

                filter_mb_edgech( &img_cr[2*2*uvlinesize], uvlinesize, bS3, qpc, a, b, h, 0);

            }

        }

        return;

    } else {

        LOCAL_ALIGNED_8(int16_t, bS, [2], [4][4]);

        int edges;

        if( IS_8x8DCT(mb_type) && (h->cbp&7) == 7 && !chroma444 ) {

            edges = 4;

            AV_WN64A(bS[0][0], 0x0002000200020002ULL);

            AV_WN64A(bS[0][2], 0x0002000200020002ULL);

            AV_WN64A(bS[1][0], 0x0002000200020002ULL);

            AV_WN64A(bS[1][2], 0x0002000200020002ULL);

        } else {

            int mask_edge1 = (3*(((5*mb_type)>>5)&1)) | (mb_type>>4); //(mb_type & (MB_TYPE_16x16 | MB_TYPE_8x16)) ? 3 : (mb_type & MB_TYPE_16x8) ? 1 : 0;

            int mask_edge0 = 3*((mask_edge1>>1) & ((5*left_type)>>5)&1); // (mb_type & (MB_TYPE_16x16 | MB_TYPE_8x16)) && (h->left_type[LTOP] & (MB_TYPE_16x16 | MB_TYPE_8x16)) ? 3 : 0;

            int step =  1+(mb_type>>24); //IS_8x8DCT(mb_type) ? 2 : 1;

            edges = 4 - 3*((mb_type>>3) & !(h->cbp & 15)); //(mb_type & MB_TYPE_16x16) && !(h->cbp & 15) ? 1 : 4;

            h->h264dsp.h264_loop_filter_strength( bS, h->non_zero_count_cache, h->ref_cache, h->mv_cache,

                                              h->list_count==2, edges, step, mask_edge0, mask_edge1, FIELD_PICTURE(h));

        }

        if( IS_INTRA(left_type) )

            AV_WN64A(bS[0][0], 0x0004000400040004ULL);

        if( IS_INTRA(top_type) )

            AV_WN64A(bS[1][0], FIELD_PICTURE(h) ? 0x0003000300030003ULL : 0x0004000400040004ULL);



#define FILTER(hv,dir,edge,intra)\

        if(AV_RN64A(bS[dir][edge])) {                                   \

            filter_mb_edge##hv( &img_y[4*edge*(dir?linesize:1<<pixel_shift)], linesize, bS[dir][edge], edge ? qp : qp##dir, a, b, h, intra );\

            if(chroma){\

                if(chroma444){\

                    filter_mb_edge##hv( &img_cb[4*edge*(dir?linesize:1<<pixel_shift)], linesize, bS[dir][edge], edge ? qpc : qpc##dir, a, b, h, intra );\

                    filter_mb_edge##hv( &img_cr[4*edge*(dir?linesize:1<<pixel_shift)], linesize, bS[dir][edge], edge ? qpc : qpc##dir, a, b, h, intra );\

                } else if(!(edge&1)) {\

                    filter_mb_edgec##hv( &img_cb[2*edge*(dir?uvlinesize:1<<pixel_shift)], uvlinesize, bS[dir][edge], edge ? qpc : qpc##dir, a, b, h, intra );\

                    filter_mb_edgec##hv( &img_cr[2*edge*(dir?uvlinesize:1<<pixel_shift)], uvlinesize, bS[dir][edge], edge ? qpc : qpc##dir, a, b, h, intra );\

                }\

            }\

        }

        if(left_type)

            FILTER(v,0,0,1);

        if( edges == 1 ) {

            if(top_type)

                FILTER(h,1,0,1);

        } else if( IS_8x8DCT(mb_type) ) {

            FILTER(v,0,2,0);

            if(top_type)

                FILTER(h,1,0,1);

            FILTER(h,1,2,0);

        } else {

            FILTER(v,0,1,0);

            FILTER(v,0,2,0);

            FILTER(v,0,3,0);

            if(top_type)

                FILTER(h,1,0,1);

            FILTER(h,1,1,0);

            FILTER(h,1,2,0);

            FILTER(h,1,3,0);

        }

#undef FILTER

    }

}
