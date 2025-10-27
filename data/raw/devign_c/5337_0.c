void MPV_decode_mb_internal(MpegEncContext *s, DCTELEM block[12][64],

                            int lowres_flag, int is_mpeg12)

{

    int mb_x, mb_y;

    const int mb_xy = s->mb_y * s->mb_stride + s->mb_x;

#if CONFIG_MPEG_XVMC_DECODER

    if(s->avctx->xvmc_acceleration){

        ff_xvmc_decode_mb(s);//xvmc uses pblocks

        return;

    }

#endif



    mb_x = s->mb_x;

    mb_y = s->mb_y;



    if(s->avctx->debug&FF_DEBUG_DCT_COEFF) {

       /* save DCT coefficients */

       int i,j;

       DCTELEM *dct = &s->current_picture.dct_coeff[mb_xy*64*6];

       for(i=0; i<6; i++)

           for(j=0; j<64; j++)

               *dct++ = block[i][s->dsp.idct_permutation[j]];

    }



    s->current_picture.qscale_table[mb_xy]= s->qscale;



    /* update DC predictors for P macroblocks */

    if (!s->mb_intra) {

        if (!is_mpeg12 && (s->h263_pred || s->h263_aic)) {

            if(s->mbintra_table[mb_xy])

                ff_clean_intra_table_entries(s);

        } else {

            s->last_dc[0] =

            s->last_dc[1] =

            s->last_dc[2] = 128 << s->intra_dc_precision;

        }

    }

    else if (!is_mpeg12 && (s->h263_pred || s->h263_aic))

        s->mbintra_table[mb_xy]=1;



    if ((s->flags&CODEC_FLAG_PSNR) || !(s->encoding && (s->intra_only || s->pict_type==FF_B_TYPE) && s->avctx->mb_decision != FF_MB_DECISION_RD)) { //FIXME precalc

        uint8_t *dest_y, *dest_cb, *dest_cr;

        int dct_linesize, dct_offset;

        op_pixels_func (*op_pix)[4];

        qpel_mc_func (*op_qpix)[16];

        const int linesize= s->current_picture.linesize[0]; //not s->linesize as this would be wrong for field pics

        const int uvlinesize= s->current_picture.linesize[1];

        const int readable= s->pict_type != FF_B_TYPE || s->encoding || s->avctx->draw_horiz_band || lowres_flag;

        const int block_size= lowres_flag ? 8>>s->avctx->lowres : 8;



        /* avoid copy if macroblock skipped in last frame too */

        /* skip only during decoding as we might trash the buffers during encoding a bit */

        if(!s->encoding){

            uint8_t *mbskip_ptr = &s->mbskip_table[mb_xy];

            const int age= s->current_picture.age;



            assert(age);



            if (s->mb_skipped) {

                s->mb_skipped= 0;

                assert(s->pict_type!=FF_I_TYPE);



                (*mbskip_ptr) ++; /* indicate that this time we skipped it */

                if(*mbskip_ptr >99) *mbskip_ptr= 99;



                /* if previous was skipped too, then nothing to do !  */

                if (*mbskip_ptr >= age && s->current_picture.reference){

                    return;

                }

            } else if(!s->current_picture.reference){

                (*mbskip_ptr) ++; /* increase counter so the age can be compared cleanly */

                if(*mbskip_ptr >99) *mbskip_ptr= 99;

            } else{

                *mbskip_ptr = 0; /* not skipped */

            }

        }



        dct_linesize = linesize << s->interlaced_dct;

        dct_offset =(s->interlaced_dct)? linesize : linesize*block_size;



        if(readable){

            dest_y=  s->dest[0];

            dest_cb= s->dest[1];

            dest_cr= s->dest[2];

        }else{

            dest_y = s->b_scratchpad;

            dest_cb= s->b_scratchpad+16*linesize;

            dest_cr= s->b_scratchpad+32*linesize;

        }



        if (!s->mb_intra) {

            /* motion handling */

            /* decoding or more than one mb_type (MC was already done otherwise) */

            if(!s->encoding){

                if(lowres_flag){

                    h264_chroma_mc_func *op_pix = s->dsp.put_h264_chroma_pixels_tab;



                    if (s->mv_dir & MV_DIR_FORWARD) {

                        MPV_motion_lowres(s, dest_y, dest_cb, dest_cr, 0, s->last_picture.data, op_pix);

                        op_pix = s->dsp.avg_h264_chroma_pixels_tab;

                    }

                    if (s->mv_dir & MV_DIR_BACKWARD) {

                        MPV_motion_lowres(s, dest_y, dest_cb, dest_cr, 1, s->next_picture.data, op_pix);

                    }

                }else{

                    op_qpix= s->me.qpel_put;

                    if ((!s->no_rounding) || s->pict_type==FF_B_TYPE){

                        op_pix = s->dsp.put_pixels_tab;

                    }else{

                        op_pix = s->dsp.put_no_rnd_pixels_tab;

                    }

                    if (s->mv_dir & MV_DIR_FORWARD) {

                        MPV_motion(s, dest_y, dest_cb, dest_cr, 0, s->last_picture.data, op_pix, op_qpix);

                        op_pix = s->dsp.avg_pixels_tab;

                        op_qpix= s->me.qpel_avg;

                    }

                    if (s->mv_dir & MV_DIR_BACKWARD) {

                        MPV_motion(s, dest_y, dest_cb, dest_cr, 1, s->next_picture.data, op_pix, op_qpix);

                    }

                }

            }



            /* skip dequant / idct if we are really late ;) */

            if(s->hurry_up>1) goto skip_idct;

            if(s->avctx->skip_idct){

                if(  (s->avctx->skip_idct >= AVDISCARD_NONREF && s->pict_type == FF_B_TYPE)

                   ||(s->avctx->skip_idct >= AVDISCARD_NONKEY && s->pict_type != FF_I_TYPE)

                   || s->avctx->skip_idct >= AVDISCARD_ALL)

                    goto skip_idct;

            }



            /* add dct residue */

            if(s->encoding || !(   s->h263_msmpeg4 || s->codec_id==CODEC_ID_MPEG1VIDEO || s->codec_id==CODEC_ID_MPEG2VIDEO

                                || (s->codec_id==CODEC_ID_MPEG4 && !s->mpeg_quant))){

                add_dequant_dct(s, block[0], 0, dest_y                          , dct_linesize, s->qscale);

                add_dequant_dct(s, block[1], 1, dest_y              + block_size, dct_linesize, s->qscale);

                add_dequant_dct(s, block[2], 2, dest_y + dct_offset             , dct_linesize, s->qscale);

                add_dequant_dct(s, block[3], 3, dest_y + dct_offset + block_size, dct_linesize, s->qscale);



                if(!CONFIG_GRAY || !(s->flags&CODEC_FLAG_GRAY)){

                    if (s->chroma_y_shift){

                        add_dequant_dct(s, block[4], 4, dest_cb, uvlinesize, s->chroma_qscale);

                        add_dequant_dct(s, block[5], 5, dest_cr, uvlinesize, s->chroma_qscale);

                    }else{

                        dct_linesize >>= 1;

                        dct_offset >>=1;

                        add_dequant_dct(s, block[4], 4, dest_cb,              dct_linesize, s->chroma_qscale);

                        add_dequant_dct(s, block[5], 5, dest_cr,              dct_linesize, s->chroma_qscale);

                        add_dequant_dct(s, block[6], 6, dest_cb + dct_offset, dct_linesize, s->chroma_qscale);

                        add_dequant_dct(s, block[7], 7, dest_cr + dct_offset, dct_linesize, s->chroma_qscale);

                    }

                }

            } else if(is_mpeg12 || (s->codec_id != CODEC_ID_WMV2)){

                add_dct(s, block[0], 0, dest_y                          , dct_linesize);

                add_dct(s, block[1], 1, dest_y              + block_size, dct_linesize);

                add_dct(s, block[2], 2, dest_y + dct_offset             , dct_linesize);

                add_dct(s, block[3], 3, dest_y + dct_offset + block_size, dct_linesize);



                if(!CONFIG_GRAY || !(s->flags&CODEC_FLAG_GRAY)){

                    if(s->chroma_y_shift){//Chroma420

                        add_dct(s, block[4], 4, dest_cb, uvlinesize);

                        add_dct(s, block[5], 5, dest_cr, uvlinesize);

                    }else{

                        //chroma422

                        dct_linesize = uvlinesize << s->interlaced_dct;

                        dct_offset =(s->interlaced_dct)? uvlinesize : uvlinesize*8;



                        add_dct(s, block[4], 4, dest_cb, dct_linesize);

                        add_dct(s, block[5], 5, dest_cr, dct_linesize);

                        add_dct(s, block[6], 6, dest_cb+dct_offset, dct_linesize);

                        add_dct(s, block[7], 7, dest_cr+dct_offset, dct_linesize);

                        if(!s->chroma_x_shift){//Chroma444

                            add_dct(s, block[8], 8, dest_cb+8, dct_linesize);

                            add_dct(s, block[9], 9, dest_cr+8, dct_linesize);

                            add_dct(s, block[10], 10, dest_cb+8+dct_offset, dct_linesize);

                            add_dct(s, block[11], 11, dest_cr+8+dct_offset, dct_linesize);

                        }

                    }

                }//fi gray

            }

            else if (CONFIG_WMV2) {

                ff_wmv2_add_mb(s, block, dest_y, dest_cb, dest_cr);

            }

        } else {

            /* dct only in intra block */

            if(s->encoding || !(s->codec_id==CODEC_ID_MPEG1VIDEO || s->codec_id==CODEC_ID_MPEG2VIDEO)){

                put_dct(s, block[0], 0, dest_y                          , dct_linesize, s->qscale);

                put_dct(s, block[1], 1, dest_y              + block_size, dct_linesize, s->qscale);

                put_dct(s, block[2], 2, dest_y + dct_offset             , dct_linesize, s->qscale);

                put_dct(s, block[3], 3, dest_y + dct_offset + block_size, dct_linesize, s->qscale);



                if(!CONFIG_GRAY || !(s->flags&CODEC_FLAG_GRAY)){

                    if(s->chroma_y_shift){

                        put_dct(s, block[4], 4, dest_cb, uvlinesize, s->chroma_qscale);

                        put_dct(s, block[5], 5, dest_cr, uvlinesize, s->chroma_qscale);

                    }else{

                        dct_offset >>=1;

                        dct_linesize >>=1;

                        put_dct(s, block[4], 4, dest_cb,              dct_linesize, s->chroma_qscale);

                        put_dct(s, block[5], 5, dest_cr,              dct_linesize, s->chroma_qscale);

                        put_dct(s, block[6], 6, dest_cb + dct_offset, dct_linesize, s->chroma_qscale);

                        put_dct(s, block[7], 7, dest_cr + dct_offset, dct_linesize, s->chroma_qscale);

                    }

                }

            }else{

                s->dsp.idct_put(dest_y                          , dct_linesize, block[0]);

                s->dsp.idct_put(dest_y              + block_size, dct_linesize, block[1]);

                s->dsp.idct_put(dest_y + dct_offset             , dct_linesize, block[2]);

                s->dsp.idct_put(dest_y + dct_offset + block_size, dct_linesize, block[3]);



                if(!CONFIG_GRAY || !(s->flags&CODEC_FLAG_GRAY)){

                    if(s->chroma_y_shift){

                        s->dsp.idct_put(dest_cb, uvlinesize, block[4]);

                        s->dsp.idct_put(dest_cr, uvlinesize, block[5]);

                    }else{



                        dct_linesize = uvlinesize << s->interlaced_dct;

                        dct_offset =(s->interlaced_dct)? uvlinesize : uvlinesize*8;



                        s->dsp.idct_put(dest_cb,              dct_linesize, block[4]);

                        s->dsp.idct_put(dest_cr,              dct_linesize, block[5]);

                        s->dsp.idct_put(dest_cb + dct_offset, dct_linesize, block[6]);

                        s->dsp.idct_put(dest_cr + dct_offset, dct_linesize, block[7]);

                        if(!s->chroma_x_shift){//Chroma444

                            s->dsp.idct_put(dest_cb + 8,              dct_linesize, block[8]);

                            s->dsp.idct_put(dest_cr + 8,              dct_linesize, block[9]);

                            s->dsp.idct_put(dest_cb + 8 + dct_offset, dct_linesize, block[10]);

                            s->dsp.idct_put(dest_cr + 8 + dct_offset, dct_linesize, block[11]);

                        }

                    }

                }//gray

            }

        }

skip_idct:

        if(!readable){

            s->dsp.put_pixels_tab[0][0](s->dest[0], dest_y ,   linesize,16);

            s->dsp.put_pixels_tab[s->chroma_x_shift][0](s->dest[1], dest_cb, uvlinesize,16 >> s->chroma_y_shift);

            s->dsp.put_pixels_tab[s->chroma_x_shift][0](s->dest[2], dest_cr, uvlinesize,16 >> s->chroma_y_shift);

        }

    }

}
