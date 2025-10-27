static int mpeg_decode_slice(Mpeg1Context *s1, int mb_y,

                             const uint8_t **buf, int buf_size)

{

    MpegEncContext *s = &s1->mpeg_enc_ctx;

    AVCodecContext *avctx= s->avctx;

    const int field_pic= s->picture_structure != PICT_FRAME;

    const int lowres= s->avctx->lowres;



    s->resync_mb_x=

    s->resync_mb_y= -1;



    if (mb_y >= s->mb_height){

        av_log(s->avctx, AV_LOG_ERROR, "slice below image (%d >= %d)\n", mb_y, s->mb_height);

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

    if(s->mb_x >= (unsigned)s->mb_width){

        av_log(s->avctx, AV_LOG_ERROR, "initial skip overflow\n");

        return -1;

    }



    if (avctx->hwaccel) {

        const uint8_t *buf_end, *buf_start = *buf - 4; /* include start_code */

        int start_code = -1;

        buf_end = ff_find_start_code(buf_start + 2, *buf + buf_size, &start_code);

        if (buf_end < *buf + buf_size)

            buf_end -= 4;

        s->mb_y = mb_y;

        if (avctx->hwaccel->decode_slice(avctx, buf_start, buf_end - buf_start) < 0)

            return DECODE_SLICE_ERROR;

        *buf = buf_end;

        return DECODE_SLICE_OK;

    }



    s->resync_mb_x= s->mb_x;

    s->resync_mb_y= s->mb_y= mb_y;

    s->mb_skip_run= 0;

    ff_init_block_index(s);



    if (s->mb_y==0 && s->mb_x==0 && (s->first_field || s->picture_structure==PICT_FRAME)) {

        if(s->avctx->debug&FF_DEBUG_PICT_INFO){

             av_log(s->avctx, AV_LOG_DEBUG, "qp:%d fc:%2d%2d%2d%2d %s %s %s %s %s dc:%d pstruct:%d fdct:%d cmv:%d qtype:%d ivlc:%d rff:%d %s\n",

                 s->qscale, s->mpeg_f_code[0][0],s->mpeg_f_code[0][1],s->mpeg_f_code[1][0],s->mpeg_f_code[1][1],

                 s->pict_type == FF_I_TYPE ? "I" : (s->pict_type == FF_P_TYPE ? "P" : (s->pict_type == FF_B_TYPE ? "B" : "S")),

                 s->progressive_sequence ? "ps" :"", s->progressive_frame ? "pf" : "", s->alternate_scan ? "alt" :"", s->top_field_first ? "top" :"",

                 s->intra_dc_precision, s->picture_structure, s->frame_pred_frame_dct, s->concealment_motion_vectors,

                 s->q_scale_type, s->intra_vlc_format, s->repeat_first_field, s->chroma_420_type ? "420" :"");

        }

    }



    for(;;) {

        //If 1, we memcpy blocks in xvmcvideo.

        if(CONFIG_MPEG_XVMC_DECODER && s->avctx->xvmc_acceleration > 1)

            ff_xvmc_init_block(s);//set s->block



        if(mpeg_decode_mb(s, s->block) < 0)

            return -1;



        if(s->current_picture.motion_val[0] && !s->encoding){ //note motion_val is normally NULL unless we want to extract the MVs

            const int wrap = s->b8_stride;

            int xy = s->mb_x*2 + s->mb_y*2*wrap;

            int motion_x, motion_y, dir, i;



            for(i=0; i<2; i++){

                for(dir=0; dir<2; dir++){

                    if (s->mb_intra || (dir==1 && s->pict_type != FF_B_TYPE)) {

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

                    assert(s->field_select[dir][i]==0 || s->field_select[dir][i]==1);

                }

                xy += wrap;

            }

        }



        s->dest[0] += 16 >> lowres;

        s->dest[1] +=(16 >> lowres) >> s->chroma_x_shift;

        s->dest[2] +=(16 >> lowres) >> s->chroma_x_shift;



        MPV_decode_mb(s, s->block);



        if (++s->mb_x >= s->mb_width) {

            const int mb_size= 16>>s->avctx->lowres;



            ff_draw_horiz_band(s, mb_size*(s->mb_y>>field_pic), mb_size);



            s->mb_x = 0;

            s->mb_y += 1<<field_pic;



            if(s->mb_y >= s->mb_height){

                int left= get_bits_left(&s->gb);

                int is_d10= s->chroma_format==2 && s->pict_type==FF_I_TYPE && avctx->profile==0 && avctx->level==5

                            && s->intra_dc_precision == 2 && s->q_scale_type == 1 && s->alternate_scan == 0

                            && s->progressive_frame == 0 /* vbv_delay == 0xBBB || 0xE10*/;



                if(left < 0 || (left && show_bits(&s->gb, FFMIN(left, 23)) && !is_d10)

                   || (avctx->error_recognition >= FF_ER_AGGRESSIVE && left>8)){

                    av_log(avctx, AV_LOG_ERROR, "end mismatch left=%d %0X\n", left, show_bits(&s->gb, FFMIN(left, 23)));

                    return -1;

                }else

                    goto eos;

            }



            ff_init_block_index(s);

        }



        /* skip mb handling */

        if (s->mb_skip_run == -1) {

            /* read increment again */

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

            if(s->mb_skip_run){

                int i;

                if(s->pict_type == FF_I_TYPE){

                    av_log(s->avctx, AV_LOG_ERROR, "skipped MB in I frame at %d %d\n", s->mb_x, s->mb_y);

                    return -1;

                }



                /* skip mb */

                s->mb_intra = 0;

                for(i=0;i<12;i++)

                    s->block_last_index[i] = -1;

                if(s->picture_structure == PICT_FRAME)

                    s->mv_type = MV_TYPE_16X16;

                else

                    s->mv_type = MV_TYPE_FIELD;

                if (s->pict_type == FF_P_TYPE) {

                    /* if P type, zero motion vector is implied */

                    s->mv_dir = MV_DIR_FORWARD;

                    s->mv[0][0][0] = s->mv[0][0][1] = 0;

                    s->last_mv[0][0][0] = s->last_mv[0][0][1] = 0;

                    s->last_mv[0][1][0] = s->last_mv[0][1][1] = 0;

                    s->field_select[0][0]= (s->picture_structure - 1) & 1;

                } else {

                    /* if B type, reuse previous vectors and directions */

                    s->mv[0][0][0] = s->last_mv[0][0][0];

                    s->mv[0][0][1] = s->last_mv[0][0][1];

                    s->mv[1][0][0] = s->last_mv[1][0][0];

                    s->mv[1][0][1] = s->last_mv[1][0][1];

                }

            }

        }

    }

eos: // end of slice

    *buf += (get_bits_count(&s->gb)-1)/8;

//printf("y %d %d %d %d\n", s->resync_mb_x, s->resync_mb_y, s->mb_x, s->mb_y);

    return 0;

}
