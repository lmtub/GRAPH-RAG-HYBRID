static int mpeg_decode_slice(Mpeg1Context *s1, int mb_y,

                             const uint8_t **buf, int buf_size)

{

    MpegEncContext *s = &s1->mpeg_enc_ctx;

    AVCodecContext *avctx= s->avctx;

    int ret;

    const int field_pic= s->picture_structure != PICT_FRAME;



    s->resync_mb_x=

    s->resync_mb_y= -1;



    if (mb_y >= s->mb_height){

        av_log(s->avctx, AV_LOG_ERROR, "slice below image (%d >= %d)\n", s->mb_y, s->mb_height);

        return -1;

    }

    

    init_get_bits(&s->gb, *buf, buf_size*8);



    ff_mpeg1_clean_buffers(s);

    s->interlaced_dct = 0;



    s->qscale = get_qscale(s);



    if(s->qscale == 0){

        av_log(s->avctx, AV_LOG_ERROR, "qscale == 0\n");

        return -1;

    }

    

    /* extra slice info */

    while (get_bits1(&s->gb) != 0) {

        skip_bits(&s->gb, 8);

    }

    

    s->mb_x=0;



    for(;;) {

        int code = get_vlc2(&s->gb, mbincr_vlc.table, MBINCR_VLC_BITS, 2);

        if (code < 0){

            av_log(s->avctx, AV_LOG_ERROR, "first mb_incr damaged\n");

            return -1;

        }

        if (code >= 33) {

            if (code == 33) {

                s->mb_x += 33;

            }

            /* otherwise, stuffing, nothing to do */

        } else {

            s->mb_x += code;

            break;

        }

    }



    s->resync_mb_x= s->mb_x;

    s->resync_mb_y= s->mb_y= mb_y;

    s->mb_skip_run= 0;

    ff_init_block_index(s);



    if (s->mb_y==0 && s->mb_x==0 && (s->first_field || s->picture_structure==PICT_FRAME)) {

        if(s->avctx->debug&FF_DEBUG_PICT_INFO){

             av_log(s->avctx, AV_LOG_DEBUG, "qp:%d fc:%2d%2d%2d%2d %s %s %s %s %s dc:%d pstruct:%d fdct:%d cmv:%d qtype:%d ivlc:%d rff:%d %s\n", 

                 s->qscale, s->mpeg_f_code[0][0],s->mpeg_f_code[0][1],s->mpeg_f_code[1][0],s->mpeg_f_code[1][1],

                 s->pict_type == I_TYPE ? "I" : (s->pict_type == P_TYPE ? "P" : (s->pict_type == B_TYPE ? "B" : "S")), 

                 s->progressive_sequence ? "ps" :"", s->progressive_frame ? "pf" : "", s->alternate_scan ? "alt" :"", s->top_field_first ? "top" :"", 

                 s->intra_dc_precision, s->picture_structure, s->frame_pred_frame_dct, s->concealment_motion_vectors,

                 s->q_scale_type, s->intra_vlc_format, s->repeat_first_field, s->chroma_420_type ? "420" :"");

        }

    }    

    

    for(;;) {

#ifdef HAVE_XVMC

        //one 1 we memcpy blocks in xvmcvideo

        if(s->avctx->xvmc_acceleration > 1)

            XVMC_init_block(s);//set s->block

#endif



	s->dsp.clear_blocks(s->block[0]);



        ret = mpeg_decode_mb(s, s->block);

        s->chroma_qscale= s->qscale;



        dprintf("ret=%d\n", ret);

        if (ret < 0)

            return -1;



        if(s->current_picture.motion_val[0] && !s->encoding){ //note motion_val is normally NULL unless we want to extract the MVs

            const int wrap = field_pic ? 2*s->b8_stride : s->b8_stride;

            int xy = s->mb_x*2 + s->mb_y*2*wrap;

            int motion_x, motion_y, dir, i;

            if(field_pic && !s->first_field)

                xy += wrap/2;



            for(i=0; i<2; i++){

                for(dir=0; dir<2; dir++){

                    if (s->mb_intra || (dir==1 && s->pict_type != B_TYPE)) {

                        motion_x = motion_y = 0;

                    }else if (s->mv_type == MV_TYPE_16X16 || (s->mv_type == MV_TYPE_FIELD && field_pic)){

                        motion_x = s->mv[dir][0][0];

                        motion_y = s->mv[dir][0][1];

                    } else /*if ((s->mv_type == MV_TYPE_FIELD) || (s->mv_type == MV_TYPE_16X8))*/ {

                        motion_x = s->mv[dir][i][0];

                        motion_y = s->mv[dir][i][1];

                    }



                    s->current_picture.motion_val[dir][xy    ][0] = motion_x;

                    s->current_picture.motion_val[dir][xy    ][1] = motion_y;

                    s->current_picture.motion_val[dir][xy + 1][0] = motion_x;

                    s->current_picture.motion_val[dir][xy + 1][1] = motion_y;

                    s->current_picture.ref_index [dir][xy    ]=

                    s->current_picture.ref_index [dir][xy + 1]= s->field_select[dir][i];

                }

                xy += wrap;

            }

        }



        s->dest[0] += 16;

        s->dest[1] += 8;

        s->dest[2] += 8;



        MPV_decode_mb(s, s->block);

        

        if (++s->mb_x >= s->mb_width) {



            ff_draw_horiz_band(s, 16*s->mb_y, 16);



            s->mb_x = 0;

            s->mb_y++;



            if(s->mb_y<<field_pic >= s->mb_height){

                int left= s->gb.size_in_bits - get_bits_count(&s->gb);



                if(left < 0 || (left && show_bits(&s->gb, FFMIN(left, 23)))

                   || (avctx->error_resilience >= FF_ER_AGGRESSIVE && left>8)){

                    av_log(avctx, AV_LOG_ERROR, "end mismatch left=%d\n", left);

                    return -1;

                }else

                    goto eos;

            }

            

            ff_init_block_index(s);

        }



        /* skip mb handling */

        if (s->mb_skip_run == -1) {

            /* read again increment */

            s->mb_skip_run = 0;

            for(;;) {

                int code = get_vlc2(&s->gb, mbincr_vlc.table, MBINCR_VLC_BITS, 2);

                if (code < 0){

                    av_log(s->avctx, AV_LOG_ERROR, "mb incr damaged\n");

                    return -1;

                }

                if (code >= 33) {

                    if (code == 33) {

                        s->mb_skip_run += 33;

                    }else if(code == 35){

                        if(s->mb_skip_run != 0 || show_bits(&s->gb, 15) != 0){

                            av_log(s->avctx, AV_LOG_ERROR, "slice mismatch\n");

                            return -1;

                        }

                        goto eos; /* end of slice */

                    }

                    /* otherwise, stuffing, nothing to do */

                } else {

                    s->mb_skip_run += code;

                    break;

                }

            }

        }

    }

eos: // end of slice

    *buf += get_bits_count(&s->gb)/8 - 1;

//printf("y %d %d %d %d\n", s->resync_mb_x, s->resync_mb_y, s->mb_x, s->mb_y);

    return 0;

}
