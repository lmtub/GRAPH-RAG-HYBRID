static void filter_mb_fast( H264Context *h, int mb_x, int mb_y, uint8_t *img_y, uint8_t *img_cb, uint8_t *img_cr, unsigned int linesize, unsigned int uvlinesize) {

    MpegEncContext * const s = &h->s;

    int mb_y_firstrow = s->picture_structure == PICT_BOTTOM_FIELD;

    int mb_xy, mb_type;

    int qp, qp0, qp1, qpc, qpc0, qpc1, qp_thresh;



    mb_xy = h->mb_xy;



    if(mb_x==0 || mb_y==mb_y_firstrow || !s->dsp.h264_loop_filter_strength || h->pps.chroma_qp_diff ||

        !(s->flags2 & CODEC_FLAG2_FAST) || //FIXME filter_mb_fast is broken, thus hasto be, but should not under CODEC_FLAG2_FAST

       (h->deblocking_filter == 2 && (h->slice_table[mb_xy] != h->slice_table[h->top_mb_xy] ||

                                      h->slice_table[mb_xy] != h->slice_table[mb_xy - 1]))) {

        filter_mb(h, mb_x, mb_y, img_y, img_cb, img_cr, linesize, uvlinesize);

        return;

    }

    assert(!FRAME_MBAFF);



    mb_type = s->current_picture.mb_type[mb_xy];

    qp = s->current_picture.qscale_table[mb_xy];

    qp0 = s->current_picture.qscale_table[mb_xy-1];

    qp1 = s->current_picture.qscale_table[h->top_mb_xy];

    qpc = get_chroma_qp( h, 0, qp );

    qpc0 = get_chroma_qp( h, 0, qp0 );

    qpc1 = get_chroma_qp( h, 0, qp1 );

    qp0 = (qp + qp0 + 1) >> 1;

    qp1 = (qp + qp1 + 1) >> 1;

    qpc0 = (qpc + qpc0 + 1) >> 1;

    qpc1 = (qpc + qpc1 + 1) >> 1;

    qp_thresh = 15 - h->slice_alpha_c0_offset;

    if(qp <= qp_thresh && qp0 <= qp_thresh && qp1 <= qp_thresh &&

       qpc <= qp_thresh && qpc0 <= qp_thresh && qpc1 <= qp_thresh)

        return;



    if( IS_INTRA(mb_type) ) {

        int16_t bS4[4] = {4,4,4,4};

        int16_t bS3[4] = {3,3,3,3};

        int16_t *bSH = FIELD_PICTURE ? bS3 : bS4;

        if( IS_8x8DCT(mb_type) ) {

            filter_mb_edgev( h, &img_y[4*0], linesize, bS4, qp0 );

            filter_mb_edgev( h, &img_y[4*2], linesize, bS3, qp );

            filter_mb_edgeh( h, &img_y[4*0*linesize], linesize, bSH, qp1 );

            filter_mb_edgeh( h, &img_y[4*2*linesize], linesize, bS3, qp );

        } else {

            filter_mb_edgev( h, &img_y[4*0], linesize, bS4, qp0 );

            filter_mb_edgev( h, &img_y[4*1], linesize, bS3, qp );

            filter_mb_edgev( h, &img_y[4*2], linesize, bS3, qp );

            filter_mb_edgev( h, &img_y[4*3], linesize, bS3, qp );

            filter_mb_edgeh( h, &img_y[4*0*linesize], linesize, bSH, qp1 );

            filter_mb_edgeh( h, &img_y[4*1*linesize], linesize, bS3, qp );

            filter_mb_edgeh( h, &img_y[4*2*linesize], linesize, bS3, qp );

            filter_mb_edgeh( h, &img_y[4*3*linesize], linesize, bS3, qp );

        }

        filter_mb_edgecv( h, &img_cb[2*0], uvlinesize, bS4, qpc0 );

        filter_mb_edgecv( h, &img_cb[2*2], uvlinesize, bS3, qpc );

        filter_mb_edgecv( h, &img_cr[2*0], uvlinesize, bS4, qpc0 );

        filter_mb_edgecv( h, &img_cr[2*2], uvlinesize, bS3, qpc );

        filter_mb_edgech( h, &img_cb[2*0*uvlinesize], uvlinesize, bSH, qpc1 );

        filter_mb_edgech( h, &img_cb[2*2*uvlinesize], uvlinesize, bS3, qpc );

        filter_mb_edgech( h, &img_cr[2*0*uvlinesize], uvlinesize, bSH, qpc1 );

        filter_mb_edgech( h, &img_cr[2*2*uvlinesize], uvlinesize, bS3, qpc );

        return;

    } else {

        DECLARE_ALIGNED_8(int16_t, bS[2][4][4]);

        uint64_t (*bSv)[4] = (uint64_t(*)[4])bS;

        int edges;

        if( IS_8x8DCT(mb_type) && (h->cbp&7) == 7 ) {

            edges = 4;

            bSv[0][0] = bSv[0][2] = bSv[1][0] = bSv[1][2] = 0x0002000200020002ULL;

        } else {

            int mask_edge1 = (mb_type & (MB_TYPE_16x16 | MB_TYPE_8x16)) ? 3 :

                             (mb_type & MB_TYPE_16x8) ? 1 : 0;

            int mask_edge0 = (mb_type & (MB_TYPE_16x16 | MB_TYPE_8x16))

                             && (s->current_picture.mb_type[mb_xy-1] & (MB_TYPE_16x16 | MB_TYPE_8x16))

                             ? 3 : 0;

            int step = IS_8x8DCT(mb_type) ? 2 : 1;

            edges = (mb_type & MB_TYPE_16x16) && !(h->cbp & 15) ? 1 : 4;

            s->dsp.h264_loop_filter_strength( bS, h->non_zero_count_cache, h->ref_cache, h->mv_cache,

                                              (h->slice_type_nos == FF_B_TYPE), edges, step, mask_edge0, mask_edge1, FIELD_PICTURE);

        }

        if( IS_INTRA(s->current_picture.mb_type[mb_xy-1]) )

            bSv[0][0] = 0x0004000400040004ULL;

        if( IS_INTRA(s->current_picture.mb_type[h->top_mb_xy]) )

            bSv[1][0] = FIELD_PICTURE ? 0x0003000300030003ULL : 0x0004000400040004ULL;



#define FILTER(hv,dir,edge)\

        if(bSv[dir][edge]) {\

            filter_mb_edge##hv( h, &img_y[4*edge*(dir?linesize:1)], linesize, bS[dir][edge], edge ? qp : qp##dir );\

            if(!(edge&1)) {\

                filter_mb_edgec##hv( h, &img_cb[2*edge*(dir?uvlinesize:1)], uvlinesize, bS[dir][edge], edge ? qpc : qpc##dir );\

                filter_mb_edgec##hv( h, &img_cr[2*edge*(dir?uvlinesize:1)], uvlinesize, bS[dir][edge], edge ? qpc : qpc##dir );\

            }\

        }

        if( edges == 1 ) {

            FILTER(v,0,0);

            FILTER(h,1,0);

        } else if( IS_8x8DCT(mb_type) ) {

            FILTER(v,0,0);

            FILTER(v,0,2);

            FILTER(h,1,0);

            FILTER(h,1,2);

        } else {

            FILTER(v,0,0);

            FILTER(v,0,1);

            FILTER(v,0,2);

            FILTER(v,0,3);

            FILTER(h,1,0);

            FILTER(h,1,1);

            FILTER(h,1,2);

            FILTER(h,1,3);

        }

#undef FILTER

    }

}
