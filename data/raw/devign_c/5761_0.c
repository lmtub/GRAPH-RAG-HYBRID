static av_always_inline void hl_decode_mb_internal(H264Context *h, int simple){

    MpegEncContext * const s = &h->s;

    const int mb_x= s->mb_x;

    const int mb_y= s->mb_y;

    const int mb_xy= h->mb_xy;

    const int mb_type= s->current_picture.mb_type[mb_xy];

    uint8_t  *dest_y, *dest_cb, *dest_cr;

    int linesize, uvlinesize /*dct_offset*/;

    int i;

    int *block_offset = &h->block_offset[0];

    const int transform_bypass = !simple && (s->qscale == 0 && h->sps.transform_bypass);

    const int is_h264 = simple || s->codec_id == CODEC_ID_H264;

    void (*idct_add)(uint8_t *dst, DCTELEM *block, int stride);

    void (*idct_dc_add)(uint8_t *dst, DCTELEM *block, int stride);



    dest_y  = s->current_picture.data[0] + (mb_x + mb_y * s->linesize  ) * 16;

    dest_cb = s->current_picture.data[1] + (mb_x + mb_y * s->uvlinesize) * 8;

    dest_cr = s->current_picture.data[2] + (mb_x + mb_y * s->uvlinesize) * 8;



    s->dsp.prefetch(dest_y + (s->mb_x&3)*4*s->linesize + 64, s->linesize, 4);

    s->dsp.prefetch(dest_cb + (s->mb_x&7)*s->uvlinesize + 64, dest_cr - dest_cb, 2);



    if (!simple && MB_FIELD) {

        linesize   = h->mb_linesize   = s->linesize * 2;

        uvlinesize = h->mb_uvlinesize = s->uvlinesize * 2;

        block_offset = &h->block_offset[24];

        if(mb_y&1){ //FIXME move out of this function?

            dest_y -= s->linesize*15;

            dest_cb-= s->uvlinesize*7;

            dest_cr-= s->uvlinesize*7;

        }

        if(FRAME_MBAFF) {

            int list;

            for(list=0; list<h->list_count; list++){

                if(!USES_LIST(mb_type, list))

                    continue;

                if(IS_16X16(mb_type)){

                    int8_t *ref = &h->ref_cache[list][scan8[0]];

                    fill_rectangle(ref, 4, 4, 8, (16+*ref)^(s->mb_y&1), 1);

                }else{

                    for(i=0; i<16; i+=4){

                        int ref = h->ref_cache[list][scan8[i]];

                        if(ref >= 0)

                            fill_rectangle(&h->ref_cache[list][scan8[i]], 2, 2, 8, (16+ref)^(s->mb_y&1), 1);

                    }

                }

            }

        }

    } else {

        linesize   = h->mb_linesize   = s->linesize;

        uvlinesize = h->mb_uvlinesize = s->uvlinesize;

//        dct_offset = s->linesize * 16;

    }



    if (!simple && IS_INTRA_PCM(mb_type)) {

        for (i=0; i<16; i++) {

            memcpy(dest_y + i*  linesize, h->mb       + i*8, 16);

        }

        for (i=0; i<8; i++) {

            memcpy(dest_cb+ i*uvlinesize, h->mb + 128 + i*4,  8);

            memcpy(dest_cr+ i*uvlinesize, h->mb + 160 + i*4,  8);

        }

    } else {

        if(IS_INTRA(mb_type)){

            if(h->deblocking_filter)

                xchg_mb_border(h, dest_y, dest_cb, dest_cr, linesize, uvlinesize, 1, simple);



            if(simple || !ENABLE_GRAY || !(s->flags&CODEC_FLAG_GRAY)){

                h->hpc.pred8x8[ h->chroma_pred_mode ](dest_cb, uvlinesize);

                h->hpc.pred8x8[ h->chroma_pred_mode ](dest_cr, uvlinesize);

            }



            if(IS_INTRA4x4(mb_type)){

                if(simple || !s->encoding){

                    if(IS_8x8DCT(mb_type)){

                        if(transform_bypass){

                            idct_dc_add =

                            idct_add    = s->dsp.add_pixels8;

                        }else if(IS_8x8DCT(mb_type)){

                            idct_dc_add = s->dsp.h264_idct8_dc_add;

                            idct_add    = s->dsp.h264_idct8_add;

                        }

                        for(i=0; i<16; i+=4){

                            uint8_t * const ptr= dest_y + block_offset[i];

                            const int dir= h->intra4x4_pred_mode_cache[ scan8[i] ];

                            if(transform_bypass && h->sps.profile_idc==244 && dir<=1){

                                h->hpc.pred8x8l_add[dir](ptr, h->mb + i*16, linesize);

                            }else{

                                const int nnz = h->non_zero_count_cache[ scan8[i] ];

                                h->hpc.pred8x8l[ dir ](ptr, (h->topleft_samples_available<<i)&0x8000,

                                                            (h->topright_samples_available<<i)&0x4000, linesize);

                                if(nnz){

                                    if(nnz == 1 && h->mb[i*16])

                                        idct_dc_add(ptr, h->mb + i*16, linesize);

                                    else

                                        idct_add   (ptr, h->mb + i*16, linesize);

                                }

                            }

                        }

                    }else{

                        if(transform_bypass){

                            idct_dc_add =

                            idct_add    = s->dsp.add_pixels4;

                        }else{

                            idct_dc_add = s->dsp.h264_idct_dc_add;

                            idct_add    = s->dsp.h264_idct_add;

                        }

                    for(i=0; i<16; i++){

                        uint8_t * const ptr= dest_y + block_offset[i];

                        const int dir= h->intra4x4_pred_mode_cache[ scan8[i] ];



                        if(transform_bypass && h->sps.profile_idc==244 && dir<=1){

                            h->hpc.pred4x4_add[dir](ptr, h->mb + i*16, linesize);

                        }else{

                            uint8_t *topright;

                            int nnz, tr;

                            if(dir == DIAG_DOWN_LEFT_PRED || dir == VERT_LEFT_PRED){

                                const int topright_avail= (h->topright_samples_available<<i)&0x8000;

                                assert(mb_y || linesize <= block_offset[i]);

                                if(!topright_avail){

                                    tr= ptr[3 - linesize]*0x01010101;

                                    topright= (uint8_t*) &tr;

                                }else

                                    topright= ptr + 4 - linesize;

                            }else

                                topright= NULL;



                            h->hpc.pred4x4[ dir ](ptr, topright, linesize);

                            nnz = h->non_zero_count_cache[ scan8[i] ];

                            if(nnz){

                                if(is_h264){

                                    if(nnz == 1 && h->mb[i*16])

                                        idct_dc_add(ptr, h->mb + i*16, linesize);

                                    else

                                        idct_add   (ptr, h->mb + i*16, linesize);

                                }else

                                    svq3_add_idct_c(ptr, h->mb + i*16, linesize, s->qscale, 0);

                            }

                        }

                    }

                    }

                }

            }else{

                h->hpc.pred16x16[ h->intra16x16_pred_mode ](dest_y , linesize);

                if(is_h264){

                    if(!transform_bypass)

                        h264_luma_dc_dequant_idct_c(h->mb, s->qscale, h->dequant4_coeff[0][s->qscale][0]);

                }else

                    svq3_luma_dc_dequant_idct_c(h->mb, s->qscale);

            }

            if(h->deblocking_filter)

                xchg_mb_border(h, dest_y, dest_cb, dest_cr, linesize, uvlinesize, 0, simple);

        }else if(is_h264){

            hl_motion(h, dest_y, dest_cb, dest_cr,

                      s->me.qpel_put, s->dsp.put_h264_chroma_pixels_tab,

                      s->me.qpel_avg, s->dsp.avg_h264_chroma_pixels_tab,

                      s->dsp.weight_h264_pixels_tab, s->dsp.biweight_h264_pixels_tab);

        }





        if(!IS_INTRA4x4(mb_type)){

            if(is_h264){

                if(IS_INTRA16x16(mb_type)){

                    if(transform_bypass){

                        if(h->sps.profile_idc==244 && (h->intra16x16_pred_mode==VERT_PRED8x8 || h->intra16x16_pred_mode==HOR_PRED8x8)){

                            h->hpc.pred16x16_add[h->intra16x16_pred_mode](dest_y, block_offset, h->mb, linesize);

                        }else{

                            for(i=0; i<16; i++){

                                if(h->non_zero_count_cache[ scan8[i] ] || h->mb[i*16])

                                    s->dsp.add_pixels4(dest_y + block_offset[i], h->mb + i*16, linesize);

                            }

                        }

                    }else{

                         s->dsp.h264_idct_add16intra(dest_y, block_offset, h->mb, linesize, h->non_zero_count_cache);

                    }

                }else if(h->cbp&15){

                    if(transform_bypass){

                        const int di = IS_8x8DCT(mb_type) ? 4 : 1;

                        idct_add= IS_8x8DCT(mb_type) ? s->dsp.add_pixels8 : s->dsp.add_pixels4;

                        for(i=0; i<16; i+=di){

                            if(h->non_zero_count_cache[ scan8[i] ]){

                                idct_add(dest_y + block_offset[i], h->mb + i*16, linesize);

                            }

                        }

                    }else{

                        if(IS_8x8DCT(mb_type)){

                            s->dsp.h264_idct8_add4(dest_y, block_offset, h->mb, linesize, h->non_zero_count_cache);

                        }else{

                            s->dsp.h264_idct_add16(dest_y, block_offset, h->mb, linesize, h->non_zero_count_cache);

                        }

                    }

                }

            }else{

                for(i=0; i<16; i++){

                    if(h->non_zero_count_cache[ scan8[i] ] || h->mb[i*16]){ //FIXME benchmark weird rule, & below

                        uint8_t * const ptr= dest_y + block_offset[i];

                        svq3_add_idct_c(ptr, h->mb + i*16, linesize, s->qscale, IS_INTRA(mb_type) ? 1 : 0);

                    }

                }

            }

        }



        if((simple || !ENABLE_GRAY || !(s->flags&CODEC_FLAG_GRAY)) && (h->cbp&0x30)){

            uint8_t *dest[2] = {dest_cb, dest_cr};

            if(transform_bypass){

                idct_add = idct_dc_add = s->dsp.add_pixels4;

            }else{

                idct_add = s->dsp.h264_idct_add;

                idct_dc_add = s->dsp.h264_idct_dc_add;

                chroma_dc_dequant_idct_c(h->mb + 16*16, h->chroma_qp[0], h->dequant4_coeff[IS_INTRA(mb_type) ? 1:4][h->chroma_qp[0]][0]);

                chroma_dc_dequant_idct_c(h->mb + 16*16+4*16, h->chroma_qp[1], h->dequant4_coeff[IS_INTRA(mb_type) ? 2:5][h->chroma_qp[1]][0]);

            }

            if(is_h264){

                if(transform_bypass && IS_INTRA(mb_type) && h->sps.profile_idc==244 && (h->chroma_pred_mode==VERT_PRED8x8 || h->chroma_pred_mode==HOR_PRED8x8)){

                    h->hpc.pred8x8_add[h->chroma_pred_mode](dest[0], block_offset + 16, h->mb + 16*16, uvlinesize);

                    h->hpc.pred8x8_add[h->chroma_pred_mode](dest[1], block_offset + 20, h->mb + 20*16, uvlinesize);

                }else{

                    for(i=16; i<16+8; i++){

                        if(h->non_zero_count_cache[ scan8[i] ])

                            idct_add   (dest[(i&4)>>2] + block_offset[i], h->mb + i*16, uvlinesize);

                        else if(h->mb[i*16])

                            idct_dc_add(dest[(i&4)>>2] + block_offset[i], h->mb + i*16, uvlinesize);

                    }

                }

            }else{

                for(i=16; i<16+8; i++){

                    if(h->non_zero_count_cache[ scan8[i] ] || h->mb[i*16]){

                        uint8_t * const ptr= dest[(i&4)>>2] + block_offset[i];

                        svq3_add_idct_c(ptr, h->mb + i*16, uvlinesize, chroma_qp[s->qscale + 12] - 12, 2);

                    }

                }

            }

        }

    }

    if(h->deblocking_filter) {

        backup_mb_border(h, dest_y, dest_cb, dest_cr, linesize, uvlinesize, simple);

        fill_caches(h, mb_type, 1); //FIXME don't fill stuff which isn't used by filter_mb

        h->chroma_qp[0] = get_chroma_qp(h, 0, s->current_picture.qscale_table[mb_xy]);

        h->chroma_qp[1] = get_chroma_qp(h, 1, s->current_picture.qscale_table[mb_xy]);

        if (!simple && FRAME_MBAFF) {

            filter_mb     (h, mb_x, mb_y, dest_y, dest_cb, dest_cr, linesize, uvlinesize);

        } else {

            filter_mb_fast(h, mb_x, mb_y, dest_y, dest_cb, dest_cr, linesize, uvlinesize);

        }

    }

}
