static void fill_caches(H264Context *h, int mb_type, int for_deblock){

    MpegEncContext * const s = &h->s;

    const int mb_xy= h->mb_xy;

    int topleft_xy, top_xy, topright_xy, left_xy[2];

    int topleft_type, top_type, topright_type, left_type[2];

    int * left_block;

    int topleft_partition= -1;

    int i;



    top_xy     = mb_xy  - (s->mb_stride << FIELD_PICTURE);



    //FIXME deblocking could skip the intra and nnz parts.

    if(for_deblock && (h->slice_num == 1 || h->slice_table[mb_xy] == h->slice_table[top_xy]) && !FRAME_MBAFF)

        return;



    /* Wow, what a mess, why didn't they simplify the interlacing & intra

     * stuff, I can't imagine that these complex rules are worth it. */



    topleft_xy = top_xy - 1;

    topright_xy= top_xy + 1;

    left_xy[1] = left_xy[0] = mb_xy-1;

    left_block = left_block_options[0];

    if(FRAME_MBAFF){

        const int pair_xy          = s->mb_x     + (s->mb_y & ~1)*s->mb_stride;

        const int top_pair_xy      = pair_xy     - s->mb_stride;

        const int topleft_pair_xy  = top_pair_xy - 1;

        const int topright_pair_xy = top_pair_xy + 1;

        const int topleft_mb_frame_flag  = !IS_INTERLACED(s->current_picture.mb_type[topleft_pair_xy]);

        const int top_mb_frame_flag      = !IS_INTERLACED(s->current_picture.mb_type[top_pair_xy]);

        const int topright_mb_frame_flag = !IS_INTERLACED(s->current_picture.mb_type[topright_pair_xy]);

        const int left_mb_frame_flag = !IS_INTERLACED(s->current_picture.mb_type[pair_xy-1]);

        const int curr_mb_frame_flag = !IS_INTERLACED(mb_type);

        const int bottom = (s->mb_y & 1);

        tprintf(s->avctx, "fill_caches: curr_mb_frame_flag:%d, left_mb_frame_flag:%d, topleft_mb_frame_flag:%d, top_mb_frame_flag:%d, topright_mb_frame_flag:%d\n", curr_mb_frame_flag, left_mb_frame_flag, topleft_mb_frame_flag, top_mb_frame_flag, topright_mb_frame_flag);

        if (bottom

                ? !curr_mb_frame_flag // bottom macroblock

                : (!curr_mb_frame_flag && !top_mb_frame_flag) // top macroblock

                ) {

            top_xy -= s->mb_stride;

        }

        if (bottom

                ? !curr_mb_frame_flag // bottom macroblock

                : (!curr_mb_frame_flag && !topleft_mb_frame_flag) // top macroblock

                ) {

            topleft_xy -= s->mb_stride;

        } else if(bottom && curr_mb_frame_flag && !left_mb_frame_flag) {

            topleft_xy += s->mb_stride;

            // take top left mv from the middle of the mb, as opposed to all other modes which use the bottom right partition

            topleft_partition = 0;

        }

        if (bottom

                ? !curr_mb_frame_flag // bottom macroblock

                : (!curr_mb_frame_flag && !topright_mb_frame_flag) // top macroblock

                ) {

            topright_xy -= s->mb_stride;

        }

        if (left_mb_frame_flag != curr_mb_frame_flag) {

            left_xy[1] = left_xy[0] = pair_xy - 1;

            if (curr_mb_frame_flag) {

                if (bottom) {

                    left_block = left_block_options[1];

                } else {

                    left_block= left_block_options[2];

                }

            } else {

                left_xy[1] += s->mb_stride;

                left_block = left_block_options[3];

            }

        }

    }



    h->top_mb_xy = top_xy;

    h->left_mb_xy[0] = left_xy[0];

    h->left_mb_xy[1] = left_xy[1];

    if(for_deblock){

        topleft_type = 0;

        topright_type = 0;

        top_type     = h->slice_table[top_xy     ] < 255 ? s->current_picture.mb_type[top_xy]     : 0;

        left_type[0] = h->slice_table[left_xy[0] ] < 255 ? s->current_picture.mb_type[left_xy[0]] : 0;

        left_type[1] = h->slice_table[left_xy[1] ] < 255 ? s->current_picture.mb_type[left_xy[1]] : 0;



        if(FRAME_MBAFF && !IS_INTRA(mb_type)){

            int list;

            for(list=0; list<h->list_count; list++){

                if(USES_LIST(mb_type,list)){

                    int8_t *ref = &s->current_picture.ref_index[list][h->mb2b8_xy[mb_xy]];

                    *(uint32_t*)&h->ref_cache[list][scan8[ 0]] =

                    *(uint32_t*)&h->ref_cache[list][scan8[ 2]] = pack16to32(ref[0],ref[1])*0x0101;

                    ref += h->b8_stride;

                    *(uint32_t*)&h->ref_cache[list][scan8[ 8]] =

                    *(uint32_t*)&h->ref_cache[list][scan8[10]] = pack16to32(ref[0],ref[1])*0x0101;

                }else{

                    fill_rectangle(&h-> mv_cache[list][scan8[ 0]], 4, 4, 8, 0, 4);

                    fill_rectangle(&h->ref_cache[list][scan8[ 0]], 4, 4, 8, (uint8_t)LIST_NOT_USED, 1);

                }

            }

        }

    }else{

        topleft_type = h->slice_table[topleft_xy ] == h->slice_num ? s->current_picture.mb_type[topleft_xy] : 0;

        top_type     = h->slice_table[top_xy     ] == h->slice_num ? s->current_picture.mb_type[top_xy]     : 0;

        topright_type= h->slice_table[topright_xy] == h->slice_num ? s->current_picture.mb_type[topright_xy]: 0;

        left_type[0] = h->slice_table[left_xy[0] ] == h->slice_num ? s->current_picture.mb_type[left_xy[0]] : 0;

        left_type[1] = h->slice_table[left_xy[1] ] == h->slice_num ? s->current_picture.mb_type[left_xy[1]] : 0;

    }



    if(IS_INTRA(mb_type)){

        h->topleft_samples_available=

        h->top_samples_available=

        h->left_samples_available= 0xFFFF;

        h->topright_samples_available= 0xEEEA;



        if(!IS_INTRA(top_type) && (top_type==0 || h->pps.constrained_intra_pred)){

            h->topleft_samples_available= 0xB3FF;

            h->top_samples_available= 0x33FF;

            h->topright_samples_available= 0x26EA;

        }

        for(i=0; i<2; i++){

            if(!IS_INTRA(left_type[i]) && (left_type[i]==0 || h->pps.constrained_intra_pred)){

                h->topleft_samples_available&= 0xDF5F;

                h->left_samples_available&= 0x5F5F;

            }

        }



        if(!IS_INTRA(topleft_type) && (topleft_type==0 || h->pps.constrained_intra_pred))

            h->topleft_samples_available&= 0x7FFF;



        if(!IS_INTRA(topright_type) && (topright_type==0 || h->pps.constrained_intra_pred))

            h->topright_samples_available&= 0xFBFF;



        if(IS_INTRA4x4(mb_type)){

            if(IS_INTRA4x4(top_type)){

                h->intra4x4_pred_mode_cache[4+8*0]= h->intra4x4_pred_mode[top_xy][4];

                h->intra4x4_pred_mode_cache[5+8*0]= h->intra4x4_pred_mode[top_xy][5];

                h->intra4x4_pred_mode_cache[6+8*0]= h->intra4x4_pred_mode[top_xy][6];

                h->intra4x4_pred_mode_cache[7+8*0]= h->intra4x4_pred_mode[top_xy][3];

            }else{

                int pred;

                if(!top_type || (IS_INTER(top_type) && h->pps.constrained_intra_pred))

                    pred= -1;

                else{

                    pred= 2;

                }

                h->intra4x4_pred_mode_cache[4+8*0]=

                h->intra4x4_pred_mode_cache[5+8*0]=

                h->intra4x4_pred_mode_cache[6+8*0]=

                h->intra4x4_pred_mode_cache[7+8*0]= pred;

            }

            for(i=0; i<2; i++){

                if(IS_INTRA4x4(left_type[i])){

                    h->intra4x4_pred_mode_cache[3+8*1 + 2*8*i]= h->intra4x4_pred_mode[left_xy[i]][left_block[0+2*i]];

                    h->intra4x4_pred_mode_cache[3+8*2 + 2*8*i]= h->intra4x4_pred_mode[left_xy[i]][left_block[1+2*i]];

                }else{

                    int pred;

                    if(!left_type[i] || (IS_INTER(left_type[i]) && h->pps.constrained_intra_pred))

                        pred= -1;

                    else{

                        pred= 2;

                    }

                    h->intra4x4_pred_mode_cache[3+8*1 + 2*8*i]=

                    h->intra4x4_pred_mode_cache[3+8*2 + 2*8*i]= pred;

                }

            }

        }

    }





/*

0 . T T. T T T T

1 L . .L . . . .

2 L . .L . . . .

3 . T TL . . . .

4 L . .L . . . .

5 L . .. . . . .

*/

//FIXME constraint_intra_pred & partitioning & nnz (let us hope this is just a typo in the spec)

    if(top_type){

        h->non_zero_count_cache[4+8*0]= h->non_zero_count[top_xy][4];

        h->non_zero_count_cache[5+8*0]= h->non_zero_count[top_xy][5];

        h->non_zero_count_cache[6+8*0]= h->non_zero_count[top_xy][6];

        h->non_zero_count_cache[7+8*0]= h->non_zero_count[top_xy][3];



        h->non_zero_count_cache[1+8*0]= h->non_zero_count[top_xy][9];

        h->non_zero_count_cache[2+8*0]= h->non_zero_count[top_xy][8];



        h->non_zero_count_cache[1+8*3]= h->non_zero_count[top_xy][12];

        h->non_zero_count_cache[2+8*3]= h->non_zero_count[top_xy][11];



    }else{

        h->non_zero_count_cache[4+8*0]=

        h->non_zero_count_cache[5+8*0]=

        h->non_zero_count_cache[6+8*0]=

        h->non_zero_count_cache[7+8*0]=



        h->non_zero_count_cache[1+8*0]=

        h->non_zero_count_cache[2+8*0]=



        h->non_zero_count_cache[1+8*3]=

        h->non_zero_count_cache[2+8*3]= h->pps.cabac && !IS_INTRA(mb_type) ? 0 : 64;



    }



    for (i=0; i<2; i++) {

        if(left_type[i]){

            h->non_zero_count_cache[3+8*1 + 2*8*i]= h->non_zero_count[left_xy[i]][left_block[0+2*i]];

            h->non_zero_count_cache[3+8*2 + 2*8*i]= h->non_zero_count[left_xy[i]][left_block[1+2*i]];

            h->non_zero_count_cache[0+8*1 +   8*i]= h->non_zero_count[left_xy[i]][left_block[4+2*i]];

            h->non_zero_count_cache[0+8*4 +   8*i]= h->non_zero_count[left_xy[i]][left_block[5+2*i]];

        }else{

            h->non_zero_count_cache[3+8*1 + 2*8*i]=

            h->non_zero_count_cache[3+8*2 + 2*8*i]=

            h->non_zero_count_cache[0+8*1 +   8*i]=

            h->non_zero_count_cache[0+8*4 +   8*i]= h->pps.cabac && !IS_INTRA(mb_type) ? 0 : 64;

        }

    }



    if( h->pps.cabac ) {

        // top_cbp

        if(top_type) {

            h->top_cbp = h->cbp_table[top_xy];

        } else if(IS_INTRA(mb_type)) {

            h->top_cbp = 0x1C0;

        } else {

            h->top_cbp = 0;

        }

        // left_cbp

        if (left_type[0]) {

            h->left_cbp = h->cbp_table[left_xy[0]] & 0x1f0;

        } else if(IS_INTRA(mb_type)) {

            h->left_cbp = 0x1C0;

        } else {

            h->left_cbp = 0;

        }

        if (left_type[0]) {

            h->left_cbp |= ((h->cbp_table[left_xy[0]]>>((left_block[0]&(~1))+1))&0x1) << 1;

        }

        if (left_type[1]) {

            h->left_cbp |= ((h->cbp_table[left_xy[1]]>>((left_block[2]&(~1))+1))&0x1) << 3;

        }

    }



#if 1

    if(IS_INTER(mb_type) || IS_DIRECT(mb_type)){

        int list;

        for(list=0; list<h->list_count; list++){

            if(!USES_LIST(mb_type, list) && !IS_DIRECT(mb_type) && !h->deblocking_filter){

                /*if(!h->mv_cache_clean[list]){

                    memset(h->mv_cache [list],  0, 8*5*2*sizeof(int16_t)); //FIXME clean only input? clean at all?

                    memset(h->ref_cache[list], PART_NOT_AVAILABLE, 8*5*sizeof(int8_t));

                    h->mv_cache_clean[list]= 1;

                }*/

                continue;

            }

            h->mv_cache_clean[list]= 0;



            if(USES_LIST(top_type, list)){

                const int b_xy= h->mb2b_xy[top_xy] + 3*h->b_stride;

                const int b8_xy= h->mb2b8_xy[top_xy] + h->b8_stride;

                *(uint32_t*)h->mv_cache[list][scan8[0] + 0 - 1*8]= *(uint32_t*)s->current_picture.motion_val[list][b_xy + 0];

                *(uint32_t*)h->mv_cache[list][scan8[0] + 1 - 1*8]= *(uint32_t*)s->current_picture.motion_val[list][b_xy + 1];

                *(uint32_t*)h->mv_cache[list][scan8[0] + 2 - 1*8]= *(uint32_t*)s->current_picture.motion_val[list][b_xy + 2];

                *(uint32_t*)h->mv_cache[list][scan8[0] + 3 - 1*8]= *(uint32_t*)s->current_picture.motion_val[list][b_xy + 3];

                h->ref_cache[list][scan8[0] + 0 - 1*8]=

                h->ref_cache[list][scan8[0] + 1 - 1*8]= s->current_picture.ref_index[list][b8_xy + 0];

                h->ref_cache[list][scan8[0] + 2 - 1*8]=

                h->ref_cache[list][scan8[0] + 3 - 1*8]= s->current_picture.ref_index[list][b8_xy + 1];

            }else{

                *(uint32_t*)h->mv_cache [list][scan8[0] + 0 - 1*8]=

                *(uint32_t*)h->mv_cache [list][scan8[0] + 1 - 1*8]=

                *(uint32_t*)h->mv_cache [list][scan8[0] + 2 - 1*8]=

                *(uint32_t*)h->mv_cache [list][scan8[0] + 3 - 1*8]= 0;

                *(uint32_t*)&h->ref_cache[list][scan8[0] + 0 - 1*8]= ((top_type ? LIST_NOT_USED : PART_NOT_AVAILABLE)&0xFF)*0x01010101;

            }



            for(i=0; i<2; i++){

                int cache_idx = scan8[0] - 1 + i*2*8;

                if(USES_LIST(left_type[i], list)){

                    const int b_xy= h->mb2b_xy[left_xy[i]] + 3;

                    const int b8_xy= h->mb2b8_xy[left_xy[i]] + 1;

                    *(uint32_t*)h->mv_cache[list][cache_idx  ]= *(uint32_t*)s->current_picture.motion_val[list][b_xy + h->b_stride*left_block[0+i*2]];

                    *(uint32_t*)h->mv_cache[list][cache_idx+8]= *(uint32_t*)s->current_picture.motion_val[list][b_xy + h->b_stride*left_block[1+i*2]];

                    h->ref_cache[list][cache_idx  ]= s->current_picture.ref_index[list][b8_xy + h->b8_stride*(left_block[0+i*2]>>1)];

                    h->ref_cache[list][cache_idx+8]= s->current_picture.ref_index[list][b8_xy + h->b8_stride*(left_block[1+i*2]>>1)];

                }else{

                    *(uint32_t*)h->mv_cache [list][cache_idx  ]=

                    *(uint32_t*)h->mv_cache [list][cache_idx+8]= 0;

                    h->ref_cache[list][cache_idx  ]=

                    h->ref_cache[list][cache_idx+8]= left_type[i] ? LIST_NOT_USED : PART_NOT_AVAILABLE;

                }

            }



            if((for_deblock || (IS_DIRECT(mb_type) && !h->direct_spatial_mv_pred)) && !FRAME_MBAFF)

                continue;



            if(USES_LIST(topleft_type, list)){

                const int b_xy = h->mb2b_xy[topleft_xy] + 3 + h->b_stride + (topleft_partition & 2*h->b_stride);

                const int b8_xy= h->mb2b8_xy[topleft_xy] + 1 + (topleft_partition & h->b8_stride);

                *(uint32_t*)h->mv_cache[list][scan8[0] - 1 - 1*8]= *(uint32_t*)s->current_picture.motion_val[list][b_xy];

                h->ref_cache[list][scan8[0] - 1 - 1*8]= s->current_picture.ref_index[list][b8_xy];

            }else{

                *(uint32_t*)h->mv_cache[list][scan8[0] - 1 - 1*8]= 0;

                h->ref_cache[list][scan8[0] - 1 - 1*8]= topleft_type ? LIST_NOT_USED : PART_NOT_AVAILABLE;

            }



            if(USES_LIST(topright_type, list)){

                const int b_xy= h->mb2b_xy[topright_xy] + 3*h->b_stride;

                const int b8_xy= h->mb2b8_xy[topright_xy] + h->b8_stride;

                *(uint32_t*)h->mv_cache[list][scan8[0] + 4 - 1*8]= *(uint32_t*)s->current_picture.motion_val[list][b_xy];

                h->ref_cache[list][scan8[0] + 4 - 1*8]= s->current_picture.ref_index[list][b8_xy];

            }else{

                *(uint32_t*)h->mv_cache [list][scan8[0] + 4 - 1*8]= 0;

                h->ref_cache[list][scan8[0] + 4 - 1*8]= topright_type ? LIST_NOT_USED : PART_NOT_AVAILABLE;

            }



            if((IS_SKIP(mb_type) || IS_DIRECT(mb_type)) && !FRAME_MBAFF)

                continue;



            h->ref_cache[list][scan8[5 ]+1] =

            h->ref_cache[list][scan8[7 ]+1] =

            h->ref_cache[list][scan8[13]+1] =  //FIXME remove past 3 (init somewhere else)

            h->ref_cache[list][scan8[4 ]] =

            h->ref_cache[list][scan8[12]] = PART_NOT_AVAILABLE;

            *(uint32_t*)h->mv_cache [list][scan8[5 ]+1]=

            *(uint32_t*)h->mv_cache [list][scan8[7 ]+1]=

            *(uint32_t*)h->mv_cache [list][scan8[13]+1]= //FIXME remove past 3 (init somewhere else)

            *(uint32_t*)h->mv_cache [list][scan8[4 ]]=

            *(uint32_t*)h->mv_cache [list][scan8[12]]= 0;



            if( h->pps.cabac ) {

                /* XXX beurk, Load mvd */

                if(USES_LIST(top_type, list)){

                    const int b_xy= h->mb2b_xy[top_xy] + 3*h->b_stride;

                    *(uint32_t*)h->mvd_cache[list][scan8[0] + 0 - 1*8]= *(uint32_t*)h->mvd_table[list][b_xy + 0];

                    *(uint32_t*)h->mvd_cache[list][scan8[0] + 1 - 1*8]= *(uint32_t*)h->mvd_table[list][b_xy + 1];

                    *(uint32_t*)h->mvd_cache[list][scan8[0] + 2 - 1*8]= *(uint32_t*)h->mvd_table[list][b_xy + 2];

                    *(uint32_t*)h->mvd_cache[list][scan8[0] + 3 - 1*8]= *(uint32_t*)h->mvd_table[list][b_xy + 3];

                }else{

                    *(uint32_t*)h->mvd_cache [list][scan8[0] + 0 - 1*8]=

                    *(uint32_t*)h->mvd_cache [list][scan8[0] + 1 - 1*8]=

                    *(uint32_t*)h->mvd_cache [list][scan8[0] + 2 - 1*8]=

                    *(uint32_t*)h->mvd_cache [list][scan8[0] + 3 - 1*8]= 0;

                }

                if(USES_LIST(left_type[0], list)){

                    const int b_xy= h->mb2b_xy[left_xy[0]] + 3;

                    *(uint32_t*)h->mvd_cache[list][scan8[0] - 1 + 0*8]= *(uint32_t*)h->mvd_table[list][b_xy + h->b_stride*left_block[0]];

                    *(uint32_t*)h->mvd_cache[list][scan8[0] - 1 + 1*8]= *(uint32_t*)h->mvd_table[list][b_xy + h->b_stride*left_block[1]];

                }else{

                    *(uint32_t*)h->mvd_cache [list][scan8[0] - 1 + 0*8]=

                    *(uint32_t*)h->mvd_cache [list][scan8[0] - 1 + 1*8]= 0;

                }

                if(USES_LIST(left_type[1], list)){

                    const int b_xy= h->mb2b_xy[left_xy[1]] + 3;

                    *(uint32_t*)h->mvd_cache[list][scan8[0] - 1 + 2*8]= *(uint32_t*)h->mvd_table[list][b_xy + h->b_stride*left_block[2]];

                    *(uint32_t*)h->mvd_cache[list][scan8[0] - 1 + 3*8]= *(uint32_t*)h->mvd_table[list][b_xy + h->b_stride*left_block[3]];

                }else{

                    *(uint32_t*)h->mvd_cache [list][scan8[0] - 1 + 2*8]=

                    *(uint32_t*)h->mvd_cache [list][scan8[0] - 1 + 3*8]= 0;

                }

                *(uint32_t*)h->mvd_cache [list][scan8[5 ]+1]=

                *(uint32_t*)h->mvd_cache [list][scan8[7 ]+1]=

                *(uint32_t*)h->mvd_cache [list][scan8[13]+1]= //FIXME remove past 3 (init somewhere else)

                *(uint32_t*)h->mvd_cache [list][scan8[4 ]]=

                *(uint32_t*)h->mvd_cache [list][scan8[12]]= 0;



                if(h->slice_type_nos == FF_B_TYPE){

                    fill_rectangle(&h->direct_cache[scan8[0]], 4, 4, 8, 0, 1);



                    if(IS_DIRECT(top_type)){

                        *(uint32_t*)&h->direct_cache[scan8[0] - 1*8]= 0x01010101;

                    }else if(IS_8X8(top_type)){

                        int b8_xy = h->mb2b8_xy[top_xy] + h->b8_stride;

                        h->direct_cache[scan8[0] + 0 - 1*8]= h->direct_table[b8_xy];

                        h->direct_cache[scan8[0] + 2 - 1*8]= h->direct_table[b8_xy + 1];

                    }else{

                        *(uint32_t*)&h->direct_cache[scan8[0] - 1*8]= 0;

                    }



                    if(IS_DIRECT(left_type[0]))

                        h->direct_cache[scan8[0] - 1 + 0*8]= 1;

                    else if(IS_8X8(left_type[0]))

                        h->direct_cache[scan8[0] - 1 + 0*8]= h->direct_table[h->mb2b8_xy[left_xy[0]] + 1 + h->b8_stride*(left_block[0]>>1)];

                    else

                        h->direct_cache[scan8[0] - 1 + 0*8]= 0;



                    if(IS_DIRECT(left_type[1]))

                        h->direct_cache[scan8[0] - 1 + 2*8]= 1;

                    else if(IS_8X8(left_type[1]))

                        h->direct_cache[scan8[0] - 1 + 2*8]= h->direct_table[h->mb2b8_xy[left_xy[1]] + 1 + h->b8_stride*(left_block[2]>>1)];

                    else

                        h->direct_cache[scan8[0] - 1 + 2*8]= 0;

                }

            }



            if(FRAME_MBAFF){

#define MAP_MVS\

                    MAP_F2F(scan8[0] - 1 - 1*8, topleft_type)\

                    MAP_F2F(scan8[0] + 0 - 1*8, top_type)\

                    MAP_F2F(scan8[0] + 1 - 1*8, top_type)\

                    MAP_F2F(scan8[0] + 2 - 1*8, top_type)\

                    MAP_F2F(scan8[0] + 3 - 1*8, top_type)\

                    MAP_F2F(scan8[0] + 4 - 1*8, topright_type)\

                    MAP_F2F(scan8[0] - 1 + 0*8, left_type[0])\

                    MAP_F2F(scan8[0] - 1 + 1*8, left_type[0])\

                    MAP_F2F(scan8[0] - 1 + 2*8, left_type[1])\

                    MAP_F2F(scan8[0] - 1 + 3*8, left_type[1])

                if(MB_FIELD){

#define MAP_F2F(idx, mb_type)\

                    if(!IS_INTERLACED(mb_type) && h->ref_cache[list][idx] >= 0){\

                        h->ref_cache[list][idx] <<= 1;\

                        h->mv_cache[list][idx][1] /= 2;\

                        h->mvd_cache[list][idx][1] /= 2;\

                    }

                    MAP_MVS

#undef MAP_F2F

                }else{

#define MAP_F2F(idx, mb_type)\

                    if(IS_INTERLACED(mb_type) && h->ref_cache[list][idx] >= 0){\

                        h->ref_cache[list][idx] >>= 1;\

                        h->mv_cache[list][idx][1] <<= 1;\

                        h->mvd_cache[list][idx][1] <<= 1;\

                    }

                    MAP_MVS

#undef MAP_F2F

                }

            }

        }

    }

#endif



    h->neighbor_transform_size= !!IS_8x8DCT(top_type) + !!IS_8x8DCT(left_type[0]);

}
