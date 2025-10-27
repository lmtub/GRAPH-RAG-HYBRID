static int decode_mb_cabac(H264Context *h) {

    MpegEncContext * const s = &h->s;

    const int mb_xy= s->mb_x + s->mb_y*s->mb_stride;

    int mb_type, partition_count, cbp = 0;

    int dct8x8_allowed= h->pps.transform_8x8_mode;



    s->dsp.clear_blocks(h->mb); //FIXME avoid if already clear (move after skip handlong?)



    tprintf(s->avctx, "pic:%d mb:%d/%d\n", h->frame_num, s->mb_x, s->mb_y);

    if( h->slice_type != I_TYPE && h->slice_type != SI_TYPE ) {

        int skip;

        /* a skipped mb needs the aff flag from the following mb */

        if( FRAME_MBAFF && s->mb_x==0 && (s->mb_y&1)==0 )

            predict_field_decoding_flag(h);

        if( FRAME_MBAFF && (s->mb_y&1)==1 && h->prev_mb_skipped )

            skip = h->next_mb_skipped;

        else

            skip = decode_cabac_mb_skip( h, s->mb_x, s->mb_y );

        /* read skip flags */

        if( skip ) {

            if( FRAME_MBAFF && (s->mb_y&1)==0 ){

                s->current_picture.mb_type[mb_xy] = MB_TYPE_SKIP;

                h->next_mb_skipped = decode_cabac_mb_skip( h, s->mb_x, s->mb_y+1 );

                if(h->next_mb_skipped)

                    predict_field_decoding_flag(h);

                else

                    h->mb_mbaff = h->mb_field_decoding_flag = decode_cabac_field_decoding_flag(h);

            }



            decode_mb_skip(h);



            h->cbp_table[mb_xy] = 0;

            h->chroma_pred_mode_table[mb_xy] = 0;

            h->last_qscale_diff = 0;



            return 0;



        }

    }

    if(FRAME_MBAFF){

        if( (s->mb_y&1) == 0 )

            h->mb_mbaff =

            h->mb_field_decoding_flag = decode_cabac_field_decoding_flag(h);

    }else

        h->mb_field_decoding_flag= (s->picture_structure!=PICT_FRAME);



    h->prev_mb_skipped = 0;



    compute_mb_neighbors(h);

    if( ( mb_type = decode_cabac_mb_type( h ) ) < 0 ) {

        av_log( h->s.avctx, AV_LOG_ERROR, "decode_cabac_mb_type failed\n" );

        return -1;

    }



    if( h->slice_type == B_TYPE ) {

        if( mb_type < 23 ){

            partition_count= b_mb_type_info[mb_type].partition_count;

            mb_type=         b_mb_type_info[mb_type].type;

        }else{

            mb_type -= 23;

            goto decode_intra_mb;

        }

    } else if( h->slice_type == P_TYPE ) {

        if( mb_type < 5) {

            partition_count= p_mb_type_info[mb_type].partition_count;

            mb_type=         p_mb_type_info[mb_type].type;

        } else {

            mb_type -= 5;

            goto decode_intra_mb;

        }

    } else {

       assert(h->slice_type == I_TYPE);

decode_intra_mb:

        partition_count = 0;

        cbp= i_mb_type_info[mb_type].cbp;

        h->intra16x16_pred_mode= i_mb_type_info[mb_type].pred_mode;

        mb_type= i_mb_type_info[mb_type].type;

    }

    if(MB_FIELD)

        mb_type |= MB_TYPE_INTERLACED;



    h->slice_table[ mb_xy ]= h->slice_num;



    if(IS_INTRA_PCM(mb_type)) {

        const uint8_t *ptr;

        unsigned int x, y;



        // We assume these blocks are very rare so we do not optimize it.

        // FIXME The two following lines get the bitstream position in the cabac

        // decode, I think it should be done by a function in cabac.h (or cabac.c).

        ptr= h->cabac.bytestream;

        if(h->cabac.low&0x1) ptr--;

        if(CABAC_BITS==16){

            if(h->cabac.low&0x1FF) ptr--;

        }



        // The pixels are stored in the same order as levels in h->mb array.

        for(y=0; y<16; y++){

            const int index= 4*(y&3) + 32*((y>>2)&1) + 128*(y>>3);

            for(x=0; x<16; x++){

                tprintf(s->avctx, "LUMA ICPM LEVEL (%3d)\n", *ptr);

                h->mb[index + (x&3) + 16*((x>>2)&1) + 64*(x>>3)]= *ptr++;

            }

        }

        for(y=0; y<8; y++){

            const int index= 256 + 4*(y&3) + 32*(y>>2);

            for(x=0; x<8; x++){

                tprintf(s->avctx, "CHROMA U ICPM LEVEL (%3d)\n", *ptr);

                h->mb[index + (x&3) + 16*(x>>2)]= *ptr++;

            }

        }

        for(y=0; y<8; y++){

            const int index= 256 + 64 + 4*(y&3) + 32*(y>>2);

            for(x=0; x<8; x++){

                tprintf(s->avctx, "CHROMA V ICPM LEVEL (%3d)\n", *ptr);

                h->mb[index + (x&3) + 16*(x>>2)]= *ptr++;

            }

        }



        ff_init_cabac_decoder(&h->cabac, ptr, h->cabac.bytestream_end - ptr);



        // All blocks are present

        h->cbp_table[mb_xy] = 0x1ef;

        h->chroma_pred_mode_table[mb_xy] = 0;

        // In deblocking, the quantizer is 0

        s->current_picture.qscale_table[mb_xy]= 0;

        h->chroma_qp = get_chroma_qp(h->pps.chroma_qp_index_offset, 0);

        // All coeffs are present

        memset(h->non_zero_count[mb_xy], 16, 16);

        s->current_picture.mb_type[mb_xy]= mb_type;

        return 0;

    }



    if(MB_MBAFF){

        h->ref_count[0] <<= 1;

        h->ref_count[1] <<= 1;

    }



    fill_caches(h, mb_type, 0);



    if( IS_INTRA( mb_type ) ) {

        int i, pred_mode;

        if( IS_INTRA4x4( mb_type ) ) {

            if( dct8x8_allowed && decode_cabac_mb_transform_size( h ) ) {

                mb_type |= MB_TYPE_8x8DCT;

                for( i = 0; i < 16; i+=4 ) {

                    int pred = pred_intra_mode( h, i );

                    int mode = decode_cabac_mb_intra4x4_pred_mode( h, pred );

                    fill_rectangle( &h->intra4x4_pred_mode_cache[ scan8[i] ], 2, 2, 8, mode, 1 );

                }

            } else {

                for( i = 0; i < 16; i++ ) {

                    int pred = pred_intra_mode( h, i );

                    h->intra4x4_pred_mode_cache[ scan8[i] ] = decode_cabac_mb_intra4x4_pred_mode( h, pred );



                //av_log( s->avctx, AV_LOG_ERROR, "i4x4 pred=%d mode=%d\n", pred, h->intra4x4_pred_mode_cache[ scan8[i] ] );

                }

            }

            write_back_intra_pred_mode(h);

            if( check_intra4x4_pred_mode(h) < 0 ) return -1;

        } else {

            h->intra16x16_pred_mode= check_intra_pred_mode( h, h->intra16x16_pred_mode );

            if( h->intra16x16_pred_mode < 0 ) return -1;

        }

        h->chroma_pred_mode_table[mb_xy] =

        pred_mode                        = decode_cabac_mb_chroma_pre_mode( h );



        pred_mode= check_intra_pred_mode( h, pred_mode );

        if( pred_mode < 0 ) return -1;

        h->chroma_pred_mode= pred_mode;

    } else if( partition_count == 4 ) {

        int i, j, sub_partition_count[4], list, ref[2][4];



        if( h->slice_type == B_TYPE ) {

            for( i = 0; i < 4; i++ ) {

                h->sub_mb_type[i] = decode_cabac_b_mb_sub_type( h );

                sub_partition_count[i]= b_sub_mb_type_info[ h->sub_mb_type[i] ].partition_count;

                h->sub_mb_type[i]=      b_sub_mb_type_info[ h->sub_mb_type[i] ].type;

            }

            if( IS_DIRECT(h->sub_mb_type[0] | h->sub_mb_type[1] |

                          h->sub_mb_type[2] | h->sub_mb_type[3]) ) {

                pred_direct_motion(h, &mb_type);

                if( h->ref_count[0] > 1 || h->ref_count[1] > 1 ) {

                    for( i = 0; i < 4; i++ )

                        if( IS_DIRECT(h->sub_mb_type[i]) )

                            fill_rectangle( &h->direct_cache[scan8[4*i]], 2, 2, 8, 1, 1 );

                }

            }

        } else {

            for( i = 0; i < 4; i++ ) {

                h->sub_mb_type[i] = decode_cabac_p_mb_sub_type( h );

                sub_partition_count[i]= p_sub_mb_type_info[ h->sub_mb_type[i] ].partition_count;

                h->sub_mb_type[i]=      p_sub_mb_type_info[ h->sub_mb_type[i] ].type;

            }

        }



        for( list = 0; list < h->list_count; list++ ) {

                for( i = 0; i < 4; i++ ) {

                    if(IS_DIRECT(h->sub_mb_type[i])) continue;

                    if(IS_DIR(h->sub_mb_type[i], 0, list)){

                        if( h->ref_count[list] > 1 )

                            ref[list][i] = decode_cabac_mb_ref( h, list, 4*i );

                        else

                            ref[list][i] = 0;

                    } else {

                        ref[list][i] = -1;

                    }

                                                       h->ref_cache[list][ scan8[4*i]+1 ]=

                    h->ref_cache[list][ scan8[4*i]+8 ]=h->ref_cache[list][ scan8[4*i]+9 ]= ref[list][i];

                }

        }



        if(dct8x8_allowed)

            dct8x8_allowed = get_dct8x8_allowed(h);



        for(list=0; list<h->list_count; list++){

            for(i=0; i<4; i++){

                if(IS_DIRECT(h->sub_mb_type[i])){

                    fill_rectangle(h->mvd_cache[list][scan8[4*i]], 2, 2, 8, 0, 4);

                    continue;

                }

                h->ref_cache[list][ scan8[4*i]   ]=h->ref_cache[list][ scan8[4*i]+1 ];



                if(IS_DIR(h->sub_mb_type[i], 0, list) && !IS_DIRECT(h->sub_mb_type[i])){

                    const int sub_mb_type= h->sub_mb_type[i];

                    const int block_width= (sub_mb_type & (MB_TYPE_16x16|MB_TYPE_16x8)) ? 2 : 1;

                    for(j=0; j<sub_partition_count[i]; j++){

                        int mpx, mpy;

                        int mx, my;

                        const int index= 4*i + block_width*j;

                        int16_t (* mv_cache)[2]= &h->mv_cache[list][ scan8[index] ];

                        int16_t (* mvd_cache)[2]= &h->mvd_cache[list][ scan8[index] ];

                        pred_motion(h, index, block_width, list, h->ref_cache[list][ scan8[index] ], &mpx, &mpy);



                        mx = mpx + decode_cabac_mb_mvd( h, list, index, 0 );

                        my = mpy + decode_cabac_mb_mvd( h, list, index, 1 );

                        tprintf(s->avctx, "final mv:%d %d\n", mx, my);



                        if(IS_SUB_8X8(sub_mb_type)){

                            mv_cache[ 1 ][0]=

                            mv_cache[ 8 ][0]= mv_cache[ 9 ][0]= mx;

                            mv_cache[ 1 ][1]=

                            mv_cache[ 8 ][1]= mv_cache[ 9 ][1]= my;



                            mvd_cache[ 1 ][0]=

                            mvd_cache[ 8 ][0]= mvd_cache[ 9 ][0]= mx - mpx;

                            mvd_cache[ 1 ][1]=

                            mvd_cache[ 8 ][1]= mvd_cache[ 9 ][1]= my - mpy;

                        }else if(IS_SUB_8X4(sub_mb_type)){

                            mv_cache[ 1 ][0]= mx;

                            mv_cache[ 1 ][1]= my;



                            mvd_cache[ 1 ][0]= mx - mpx;

                            mvd_cache[ 1 ][1]= my - mpy;

                        }else if(IS_SUB_4X8(sub_mb_type)){

                            mv_cache[ 8 ][0]= mx;

                            mv_cache[ 8 ][1]= my;



                            mvd_cache[ 8 ][0]= mx - mpx;

                            mvd_cache[ 8 ][1]= my - mpy;

                        }

                        mv_cache[ 0 ][0]= mx;

                        mv_cache[ 0 ][1]= my;



                        mvd_cache[ 0 ][0]= mx - mpx;

                        mvd_cache[ 0 ][1]= my - mpy;

                    }

                }else{

                    uint32_t *p= (uint32_t *)&h->mv_cache[list][ scan8[4*i] ][0];

                    uint32_t *pd= (uint32_t *)&h->mvd_cache[list][ scan8[4*i] ][0];

                    p[0] = p[1] = p[8] = p[9] = 0;

                    pd[0]= pd[1]= pd[8]= pd[9]= 0;

                }

            }

        }

    } else if( IS_DIRECT(mb_type) ) {

        pred_direct_motion(h, &mb_type);

        fill_rectangle(h->mvd_cache[0][scan8[0]], 4, 4, 8, 0, 4);

        fill_rectangle(h->mvd_cache[1][scan8[0]], 4, 4, 8, 0, 4);

        dct8x8_allowed &= h->sps.direct_8x8_inference_flag;

    } else {

        int list, mx, my, i, mpx, mpy;

        if(IS_16X16(mb_type)){

            for(list=0; list<h->list_count; list++){

                if(IS_DIR(mb_type, 0, list)){

                        const int ref = h->ref_count[list] > 1 ? decode_cabac_mb_ref( h, list, 0 ) : 0;

                        fill_rectangle(&h->ref_cache[list][ scan8[0] ], 4, 4, 8, ref, 1);

                }else

                    fill_rectangle(&h->ref_cache[list][ scan8[0] ], 4, 4, 8, (uint8_t)LIST_NOT_USED, 1); //FIXME factorize and the other fill_rect below too

            }

            for(list=0; list<h->list_count; list++){

                if(IS_DIR(mb_type, 0, list)){

                    pred_motion(h, 0, 4, list, h->ref_cache[list][ scan8[0] ], &mpx, &mpy);



                    mx = mpx + decode_cabac_mb_mvd( h, list, 0, 0 );

                    my = mpy + decode_cabac_mb_mvd( h, list, 0, 1 );

                    tprintf(s->avctx, "final mv:%d %d\n", mx, my);



                    fill_rectangle(h->mvd_cache[list][ scan8[0] ], 4, 4, 8, pack16to32(mx-mpx,my-mpy), 4);

                    fill_rectangle(h->mv_cache[list][ scan8[0] ], 4, 4, 8, pack16to32(mx,my), 4);

                }else

                    fill_rectangle(h->mv_cache[list][ scan8[0] ], 4, 4, 8, 0, 4);

            }

        }

        else if(IS_16X8(mb_type)){

            for(list=0; list<h->list_count; list++){

                    for(i=0; i<2; i++){

                        if(IS_DIR(mb_type, i, list)){

                            const int ref= h->ref_count[list] > 1 ? decode_cabac_mb_ref( h, list, 8*i ) : 0;

                            fill_rectangle(&h->ref_cache[list][ scan8[0] + 16*i ], 4, 2, 8, ref, 1);

                        }else

                            fill_rectangle(&h->ref_cache[list][ scan8[0] + 16*i ], 4, 2, 8, (LIST_NOT_USED&0xFF), 1);

                    }

            }

            for(list=0; list<h->list_count; list++){

                for(i=0; i<2; i++){

                    if(IS_DIR(mb_type, i, list)){

                        pred_16x8_motion(h, 8*i, list, h->ref_cache[list][scan8[0] + 16*i], &mpx, &mpy);

                        mx = mpx + decode_cabac_mb_mvd( h, list, 8*i, 0 );

                        my = mpy + decode_cabac_mb_mvd( h, list, 8*i, 1 );

                        tprintf(s->avctx, "final mv:%d %d\n", mx, my);



                        fill_rectangle(h->mvd_cache[list][ scan8[0] + 16*i ], 4, 2, 8, pack16to32(mx-mpx,my-mpy), 4);

                        fill_rectangle(h->mv_cache[list][ scan8[0] + 16*i ], 4, 2, 8, pack16to32(mx,my), 4);

                    }else{

                        fill_rectangle(h->mvd_cache[list][ scan8[0] + 16*i ], 4, 2, 8, 0, 4);

                        fill_rectangle(h-> mv_cache[list][ scan8[0] + 16*i ], 4, 2, 8, 0, 4);

                    }

                }

            }

        }else{

            assert(IS_8X16(mb_type));

            for(list=0; list<h->list_count; list++){

                    for(i=0; i<2; i++){

                        if(IS_DIR(mb_type, i, list)){ //FIXME optimize

                            const int ref= h->ref_count[list] > 1 ? decode_cabac_mb_ref( h, list, 4*i ) : 0;

                            fill_rectangle(&h->ref_cache[list][ scan8[0] + 2*i ], 2, 4, 8, ref, 1);

                        }else

                            fill_rectangle(&h->ref_cache[list][ scan8[0] + 2*i ], 2, 4, 8, (LIST_NOT_USED&0xFF), 1);

                    }

            }

            for(list=0; list<h->list_count; list++){

                for(i=0; i<2; i++){

                    if(IS_DIR(mb_type, i, list)){

                        pred_8x16_motion(h, i*4, list, h->ref_cache[list][ scan8[0] + 2*i ], &mpx, &mpy);

                        mx = mpx + decode_cabac_mb_mvd( h, list, 4*i, 0 );

                        my = mpy + decode_cabac_mb_mvd( h, list, 4*i, 1 );



                        tprintf(s->avctx, "final mv:%d %d\n", mx, my);

                        fill_rectangle(h->mvd_cache[list][ scan8[0] + 2*i ], 2, 4, 8, pack16to32(mx-mpx,my-mpy), 4);

                        fill_rectangle(h->mv_cache[list][ scan8[0] + 2*i ], 2, 4, 8, pack16to32(mx,my), 4);

                    }else{

                        fill_rectangle(h->mvd_cache[list][ scan8[0] + 2*i ], 2, 4, 8, 0, 4);

                        fill_rectangle(h-> mv_cache[list][ scan8[0] + 2*i ], 2, 4, 8, 0, 4);

                    }

                }

            }

        }

    }



   if( IS_INTER( mb_type ) ) {

        h->chroma_pred_mode_table[mb_xy] = 0;

        write_back_motion( h, mb_type );

   }



    if( !IS_INTRA16x16( mb_type ) ) {

        cbp  = decode_cabac_mb_cbp_luma( h );

        cbp |= decode_cabac_mb_cbp_chroma( h ) << 4;

    }



    h->cbp_table[mb_xy] = h->cbp = cbp;



    if( dct8x8_allowed && (cbp&15) && !IS_INTRA( mb_type ) ) {

        if( decode_cabac_mb_transform_size( h ) )

            mb_type |= MB_TYPE_8x8DCT;

    }

    s->current_picture.mb_type[mb_xy]= mb_type;



    if( cbp || IS_INTRA16x16( mb_type ) ) {

        const uint8_t *scan, *scan8x8, *dc_scan;

        int dqp;



        if(IS_INTERLACED(mb_type)){

            scan8x8= s->qscale ? h->field_scan8x8 : h->field_scan8x8_q0;

            scan= s->qscale ? h->field_scan : h->field_scan_q0;

            dc_scan= luma_dc_field_scan;

        }else{

            scan8x8= s->qscale ? h->zigzag_scan8x8 : h->zigzag_scan8x8_q0;

            scan= s->qscale ? h->zigzag_scan : h->zigzag_scan_q0;

            dc_scan= luma_dc_zigzag_scan;

        }



        h->last_qscale_diff = dqp = decode_cabac_mb_dqp( h );

        if( dqp == INT_MIN ){

            av_log(h->s.avctx, AV_LOG_ERROR, "cabac decode of qscale diff failed at %d %d\n", s->mb_x, s->mb_y);

            return -1;

        }

        s->qscale += dqp;

        if(((unsigned)s->qscale) > 51){

            if(s->qscale<0) s->qscale+= 52;

            else            s->qscale-= 52;

        }

        h->chroma_qp = get_chroma_qp(h->pps.chroma_qp_index_offset, s->qscale);



        if( IS_INTRA16x16( mb_type ) ) {

            int i;

            //av_log( s->avctx, AV_LOG_ERROR, "INTRA16x16 DC\n" );

            if( decode_cabac_residual( h, h->mb, 0, 0, dc_scan, NULL, 16) < 0)

                return -1;

            if( cbp&15 ) {

                for( i = 0; i < 16; i++ ) {

                    //av_log( s->avctx, AV_LOG_ERROR, "INTRA16x16 AC:%d\n", i );

                    if( decode_cabac_residual(h, h->mb + 16*i, 1, i, scan + 1, h->dequant4_coeff[0][s->qscale], 15) < 0 )

                        return -1;

                }

            } else {

                fill_rectangle(&h->non_zero_count_cache[scan8[0]], 4, 4, 8, 0, 1);

            }

        } else {

            int i8x8, i4x4;

            for( i8x8 = 0; i8x8 < 4; i8x8++ ) {

                if( cbp & (1<<i8x8) ) {

                    if( IS_8x8DCT(mb_type) ) {

                        if( decode_cabac_residual(h, h->mb + 64*i8x8, 5, 4*i8x8,

                            scan8x8, h->dequant8_coeff[IS_INTRA( mb_type ) ? 0:1][s->qscale], 64) < 0 )

                            return -1;

                    } else

                    for( i4x4 = 0; i4x4 < 4; i4x4++ ) {

                        const int index = 4*i8x8 + i4x4;

                        //av_log( s->avctx, AV_LOG_ERROR, "Luma4x4: %d\n", index );

//START_TIMER

                        if( decode_cabac_residual(h, h->mb + 16*index, 2, index, scan, h->dequant4_coeff[IS_INTRA( mb_type ) ? 0:3][s->qscale], 16) < 0 )

                            return -1;

//STOP_TIMER("decode_residual")

                    }

                } else {

                    uint8_t * const nnz= &h->non_zero_count_cache[ scan8[4*i8x8] ];

                    nnz[0] = nnz[1] = nnz[8] = nnz[9] = 0;

                }

            }

        }



        if( cbp&0x30 ){

            int c;

            for( c = 0; c < 2; c++ ) {

                //av_log( s->avctx, AV_LOG_ERROR, "INTRA C%d-DC\n",c );

                if( decode_cabac_residual(h, h->mb + 256 + 16*4*c, 3, c, chroma_dc_scan, NULL, 4) < 0)

                    return -1;

            }

        }



        if( cbp&0x20 ) {

            int c, i;

            for( c = 0; c < 2; c++ ) {

                const uint32_t *qmul = h->dequant4_coeff[c+1+(IS_INTRA( mb_type ) ? 0:3)][h->chroma_qp];

                for( i = 0; i < 4; i++ ) {

                    const int index = 16 + 4 * c + i;

                    //av_log( s->avctx, AV_LOG_ERROR, "INTRA C%d-AC %d\n",c, index - 16 );

                    if( decode_cabac_residual(h, h->mb + 16*index, 4, index - 16, scan + 1, qmul, 15) < 0)

                        return -1;

                }

            }

        } else {

            uint8_t * const nnz= &h->non_zero_count_cache[0];

            nnz[ scan8[16]+0 ] = nnz[ scan8[16]+1 ] =nnz[ scan8[16]+8 ] =nnz[ scan8[16]+9 ] =

            nnz[ scan8[20]+0 ] = nnz[ scan8[20]+1 ] =nnz[ scan8[20]+8 ] =nnz[ scan8[20]+9 ] = 0;

        }

    } else {

        uint8_t * const nnz= &h->non_zero_count_cache[0];

        fill_rectangle(&nnz[scan8[0]], 4, 4, 8, 0, 1);

        nnz[ scan8[16]+0 ] = nnz[ scan8[16]+1 ] =nnz[ scan8[16]+8 ] =nnz[ scan8[16]+9 ] =

        nnz[ scan8[20]+0 ] = nnz[ scan8[20]+1 ] =nnz[ scan8[20]+8 ] =nnz[ scan8[20]+9 ] = 0;

        h->last_qscale_diff = 0;

    }



    s->current_picture.qscale_table[mb_xy]= s->qscale;

    write_back_non_zero_count(h);



    if(MB_MBAFF){

        h->ref_count[0] >>= 1;

        h->ref_count[1] >>= 1;

    }



    return 0;

}
