static int mpeg_decode_slice(AVCodecContext *avctx, 

                              AVFrame *pict,

                              int start_code,

                              UINT8 *buf, int buf_size)

{

    Mpeg1Context *s1 = avctx->priv_data;

    MpegEncContext *s = &s1->mpeg_enc_ctx;

    int ret;



    start_code = (start_code - 1) & 0xff;

    if (start_code >= s->mb_height){

        fprintf(stderr, "slice below image (%d >= %d)\n", start_code, s->mb_height);

        return DECODE_SLICE_ERROR;

    }

    s->last_dc[0] = 1 << (7 + s->intra_dc_precision);

    s->last_dc[1] = s->last_dc[0];

    s->last_dc[2] = s->last_dc[0];

    memset(s->last_mv, 0, sizeof(s->last_mv));

    /* start frame decoding */

    if (s->first_slice) {

        s->first_slice = 0;

        if(MPV_frame_start(s, avctx) < 0)

            return DECODE_SLICE_FATAL_ERROR;

            

        if(s->avctx->debug&FF_DEBUG_PICT_INFO){

             printf("qp:%d fc:%2d%2d%2d%2d %s %s %s %s dc:%d pstruct:%d fdct:%d cmv:%d qtype:%d ivlc:%d rff:%d %s\n", 

                 s->qscale, s->mpeg_f_code[0][0],s->mpeg_f_code[0][1],s->mpeg_f_code[1][0],s->mpeg_f_code[1][1],

                 s->pict_type == I_TYPE ? "I" : (s->pict_type == P_TYPE ? "P" : (s->pict_type == B_TYPE ? "B" : "S")), 

                 s->progressive_sequence ? "pro" :"", s->alternate_scan ? "alt" :"", s->top_field_first ? "top" :"", 

                 s->intra_dc_precision, s->picture_structure, s->frame_pred_frame_dct, s->concealment_motion_vectors,

                 s->q_scale_type, s->intra_vlc_format, s->repeat_first_field, s->chroma_420_type ? "420" :"");

        }

    }



    init_get_bits(&s->gb, buf, buf_size);



    s->qscale = get_qscale(s);

    /* extra slice info */

    while (get_bits1(&s->gb) != 0) {

        skip_bits(&s->gb, 8);

    }



    s->mb_x=0;

    for(;;) {

        int code = get_vlc2(&s->gb, mbincr_vlc.table, MBINCR_VLC_BITS, 2);

        if (code < 0)

            return -1; /* error = end of slice, but empty slice is bad or?*/

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

    s->mb_y = start_code;

    s->mb_incr= 1;



    for(;;) {

	s->dsp.clear_blocks(s->block[0]);

        

        ret = mpeg_decode_mb(s, s->block);

        dprintf("ret=%d\n", ret);

        if (ret < 0)

            return -1;

    

        MPV_decode_mb(s, s->block);



        if (++s->mb_x >= s->mb_width) {

            ff_draw_horiz_band(s);



            s->mb_x = 0;

            s->mb_y++;

            PRINT_QP("%s", "\n");

        }

        PRINT_QP("%2d", s->qscale);



        /* skip mb handling */

        if (s->mb_incr == 0) {

            /* read again increment */

            s->mb_incr = 1;

            for(;;) {

                int code = get_vlc2(&s->gb, mbincr_vlc.table, MBINCR_VLC_BITS, 2);

                if (code < 0)

                    goto eos; /* error = end of slice */

                if (code >= 33) {

                    if (code == 33) {

                        s->mb_incr += 33;

                    }

                    /* otherwise, stuffing, nothing to do */

                } else {

                    s->mb_incr += code;

                    break;

                }

            }

        }

        if(s->mb_y >= s->mb_height){

            fprintf(stderr, "slice too long\n");

            return DECODE_SLICE_ERROR;

        }

    }

eos: //end of slice

    

    emms_c();



    /* end of slice reached */

    if (/*s->mb_x == 0 &&*/

        s->mb_y == s->mb_height) {

        /* end of image */



        if(s->mpeg2)

            s->qscale >>=1;



        MPV_frame_end(s);



        if (s->pict_type == B_TYPE || s->low_delay) {

            *pict= *(AVFrame*)&s->current_picture;

        } else {

            s->picture_number++;

            /* latency of 1 frame for I and P frames */

            /* XXX: use another variable than picture_number */

            if (s->last_picture.data[0] == NULL) {

                return DECODE_SLICE_OK;

            } else {

                *pict= *(AVFrame*)&s->last_picture;

            }

        }

        return DECODE_SLICE_EOP;

    } else {

        return DECODE_SLICE_OK;

    }

}
