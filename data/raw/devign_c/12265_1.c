static void encode_mb(MpegEncContext *s, int motion_x, int motion_y)

{

    const int mb_x= s->mb_x;

    const int mb_y= s->mb_y;

    int i;

#if 0

        if (s->interlaced_dct) {

            dct_linesize = s->linesize * 2;

            dct_offset = s->linesize;

        } else {

            dct_linesize = s->linesize;

            dct_offset = s->linesize * 8;

        }

#endif



    if (s->mb_intra) {

        UINT8 *ptr;

        int wrap;



        wrap = s->linesize;

        ptr = s->new_picture[0] + (mb_y * 16 * wrap) + mb_x * 16;

        get_pixels(s->block[0], ptr               , wrap);

        get_pixels(s->block[1], ptr            + 8, wrap);

        get_pixels(s->block[2], ptr + 8 * wrap    , wrap);

        get_pixels(s->block[3], ptr + 8 * wrap + 8, wrap);



        wrap >>=1;

        ptr = s->new_picture[1] + (mb_y * 8 * wrap) + mb_x * 8;

        get_pixels(s->block[4], ptr, wrap);



        ptr = s->new_picture[2] + (mb_y * 8 * wrap) + mb_x * 8;

        get_pixels(s->block[5], ptr, wrap);

    }else{

        op_pixels_func *op_pix;

        qpel_mc_func *op_qpix;

        UINT8 *dest_y, *dest_cb, *dest_cr;

        UINT8 *ptr;

        int wrap;



        dest_y  = s->current_picture[0] + (mb_y * 16 * s->linesize       ) + mb_x * 16;

        dest_cb = s->current_picture[1] + (mb_y * 8  * (s->linesize >> 1)) + mb_x * 8;

        dest_cr = s->current_picture[2] + (mb_y * 8  * (s->linesize >> 1)) + mb_x * 8;



        if ((!s->no_rounding) || s->pict_type==B_TYPE){

            op_pix = put_pixels_tab;

            op_qpix= qpel_mc_rnd_tab;

        }else{

            op_pix = put_no_rnd_pixels_tab;

            op_qpix= qpel_mc_no_rnd_tab;

        }



        if (s->mv_dir & MV_DIR_FORWARD) {

            MPV_motion(s, dest_y, dest_cb, dest_cr, 0, s->last_picture, op_pix, op_qpix);

           if ((!s->no_rounding) || s->pict_type==B_TYPE)

                op_pix = avg_pixels_tab;

            else

                op_pix = avg_no_rnd_pixels_tab;

        }

        if (s->mv_dir & MV_DIR_BACKWARD) {

            MPV_motion(s, dest_y, dest_cb, dest_cr, 1, s->next_picture, op_pix, op_qpix);

        }

        wrap = s->linesize;

        ptr = s->new_picture[0] + (mb_y * 16 * wrap) + mb_x * 16;

        diff_pixels(s->block[0], ptr               , dest_y               , wrap);

        diff_pixels(s->block[1], ptr            + 8, dest_y            + 8, wrap);

        diff_pixels(s->block[2], ptr + 8 * wrap    , dest_y + 8 * wrap    , wrap);

        diff_pixels(s->block[3], ptr + 8 * wrap + 8, dest_y + 8 * wrap + 8, wrap);



        wrap >>=1;

        ptr = s->new_picture[1] + (mb_y * 8 * wrap) + mb_x * 8;

        diff_pixels(s->block[4], ptr, dest_cb, wrap);



        ptr = s->new_picture[2] + (mb_y * 8 * wrap) + mb_x * 8;

        diff_pixels(s->block[5], ptr, dest_cr, wrap);

    }

            

#if 0

            {

                float adap_parm;

                

                adap_parm = ((s->avg_mb_var << 1) + s->mb_var[s->mb_width*mb_y+mb_x] + 1.0) /

                            ((s->mb_var[s->mb_width*mb_y+mb_x] << 1) + s->avg_mb_var + 1.0);

            

                printf("\ntype=%c qscale=%2d adap=%0.2f dquant=%4.2f var=%4d avgvar=%4d", 

                        (s->mb_type[s->mb_width*mb_y+mb_x] > 0) ? 'I' : 'P', 

                        s->qscale, adap_parm, s->qscale*adap_parm,

                        s->mb_var[s->mb_width*mb_y+mb_x], s->avg_mb_var);

            }

#endif

    /* DCT & quantize */

    if (s->h263_msmpeg4) {

        msmpeg4_dc_scale(s);

    } else if (s->h263_pred) {

        h263_dc_scale(s);

    } else {

        /* default quantization values */

        s->y_dc_scale = 8;

        s->c_dc_scale = 8;

    }

    for(i=0;i<6;i++) {

        s->block_last_index[i] = dct_quantize(s, s->block[i], i, s->qscale);

    }



    /* huffman encode */

    switch(s->out_format) {

    case FMT_MPEG1:

        mpeg1_encode_mb(s, s->block, motion_x, motion_y);

        break;

    case FMT_H263:

        if (s->h263_msmpeg4)

            msmpeg4_encode_mb(s, s->block, motion_x, motion_y);

        else if(s->h263_pred)

            mpeg4_encode_mb(s, s->block, motion_x, motion_y);

        else

            h263_encode_mb(s, s->block, motion_x, motion_y);

        break;

    case FMT_MJPEG:

        mjpeg_encode_mb(s, s->block);

        break;

    }

}
