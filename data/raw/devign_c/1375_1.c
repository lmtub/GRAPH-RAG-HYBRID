static int svq3_decode_frame(AVCodecContext *avctx,

                             void *data, int *data_size,

                             AVPacket *avpkt)

{

    SVQ3Context *svq3 = avctx->priv_data;

    H264Context *h = &svq3->h;

    MpegEncContext *s = &h->s;

    int buf_size = avpkt->size;

    int m, mb_type, left;

    uint8_t *buf;



    /* special case for last picture */

    if (buf_size == 0) {

        if (s->next_picture_ptr && !s->low_delay) {

            *(AVFrame *) data = *(AVFrame *) &s->next_picture;

            s->next_picture_ptr = NULL;

            *data_size = sizeof(AVFrame);

        }

        return 0;

    }



    s->mb_x = s->mb_y = h->mb_xy = 0;



    if (svq3->watermark_key) {

        svq3->buf = av_fast_realloc(svq3->buf, &svq3->buf_size,

                                    buf_size+FF_INPUT_BUFFER_PADDING_SIZE);

        if (!svq3->buf)

            return AVERROR(ENOMEM);

        memcpy(svq3->buf, avpkt->data, buf_size);

        buf = svq3->buf;

    } else {

        buf = avpkt->data;

    }



    init_get_bits(&s->gb, buf, 8*buf_size);



    if (svq3_decode_slice_header(avctx))

        return -1;



    s->pict_type = h->slice_type;

    s->picture_number = h->slice_num;



    if (avctx->debug&FF_DEBUG_PICT_INFO){

        av_log(h->s.avctx, AV_LOG_DEBUG, "%c hpel:%d, tpel:%d aqp:%d qp:%d, slice_num:%02X\n",

               av_get_picture_type_char(s->pict_type), svq3->halfpel_flag, svq3->thirdpel_flag,

               s->adaptive_quant, s->qscale, h->slice_num);

    }



    /* for skipping the frame */

    s->current_picture.pict_type = s->pict_type;

    s->current_picture.key_frame = (s->pict_type == AV_PICTURE_TYPE_I);



    /* Skip B-frames if we do not have reference frames. */

    if (s->last_picture_ptr == NULL && s->pict_type == AV_PICTURE_TYPE_B)

        return 0;

    if (  (avctx->skip_frame >= AVDISCARD_NONREF && s->pict_type == AV_PICTURE_TYPE_B)

        ||(avctx->skip_frame >= AVDISCARD_NONKEY && s->pict_type != AV_PICTURE_TYPE_I)

        || avctx->skip_frame >= AVDISCARD_ALL)

        return 0;



    if (s->next_p_frame_damaged) {

        if (s->pict_type == AV_PICTURE_TYPE_B)

            return 0;

        else

            s->next_p_frame_damaged = 0;

    }



    if (ff_h264_frame_start(h) < 0)

        return -1;



    if (s->pict_type == AV_PICTURE_TYPE_B) {

        h->frame_num_offset = (h->slice_num - h->prev_frame_num);



        if (h->frame_num_offset < 0) {

            h->frame_num_offset += 256;

        }

        if (h->frame_num_offset == 0 || h->frame_num_offset >= h->prev_frame_num_offset) {

            av_log(h->s.avctx, AV_LOG_ERROR, "error in B-frame picture id\n");

            return -1;

        }

    } else {

        h->prev_frame_num = h->frame_num;

        h->frame_num = h->slice_num;

        h->prev_frame_num_offset = (h->frame_num - h->prev_frame_num);



        if (h->prev_frame_num_offset < 0) {

            h->prev_frame_num_offset += 256;

        }

    }



    for (m = 0; m < 2; m++){

        int i;

        for (i = 0; i < 4; i++){

            int j;

            for (j = -1; j < 4; j++)

                h->ref_cache[m][scan8[0] + 8*i + j]= 1;

            if (i < 3)

                h->ref_cache[m][scan8[0] + 8*i + j]= PART_NOT_AVAILABLE;

        }

    }



    for (s->mb_y = 0; s->mb_y < s->mb_height; s->mb_y++) {

        for (s->mb_x = 0; s->mb_x < s->mb_width; s->mb_x++) {

            h->mb_xy = s->mb_x + s->mb_y*s->mb_stride;



            if ( (get_bits_count(&s->gb) + 7) >= s->gb.size_in_bits &&

                ((get_bits_count(&s->gb) & 7) == 0 || show_bits(&s->gb, (-get_bits_count(&s->gb) & 7)) == 0)) {



                skip_bits(&s->gb, svq3->next_slice_index - get_bits_count(&s->gb));

                s->gb.size_in_bits = 8*buf_size;



                if (svq3_decode_slice_header(avctx))

                    return -1;



                /* TODO: support s->mb_skip_run */

            }



            mb_type = svq3_get_ue_golomb(&s->gb);



            if (s->pict_type == AV_PICTURE_TYPE_I) {

                mb_type += 8;

            } else if (s->pict_type == AV_PICTURE_TYPE_B && mb_type >= 4) {

                mb_type += 4;

            }

            if (mb_type > 33 || svq3_decode_mb(svq3, mb_type)) {

                av_log(h->s.avctx, AV_LOG_ERROR, "error while decoding MB %d %d\n", s->mb_x, s->mb_y);

                return -1;

            }



            if (mb_type != 0) {

                ff_h264_hl_decode_mb (h);

            }



            if (s->pict_type != AV_PICTURE_TYPE_B && !s->low_delay) {

                s->current_picture.mb_type[s->mb_x + s->mb_y*s->mb_stride] =

                    (s->pict_type == AV_PICTURE_TYPE_P && mb_type < 8) ? (mb_type - 1) : -1;

            }

        }



        ff_draw_horiz_band(s, 16*s->mb_y, 16);

    }



    left = buf_size*8 - get_bits_count(&s->gb);



    if (s->mb_y != s->mb_height || s->mb_x != s->mb_width) {

        av_log(avctx, AV_LOG_INFO, "frame num %d incomplete pic x %d y %d left %d\n", avctx->frame_number, s->mb_y, s->mb_x, left);

        //av_hex_dump(stderr, buf+buf_size-8, 8);

    }



    if (left < 0) {

        av_log(avctx, AV_LOG_ERROR, "frame num %d left %d\n", avctx->frame_number, left);

        return -1;

    }



    MPV_frame_end(s);



    if (s->pict_type == AV_PICTURE_TYPE_B || s->low_delay) {

        *(AVFrame *) data = *(AVFrame *) &s->current_picture;

    } else {

        *(AVFrame *) data = *(AVFrame *) &s->last_picture;

    }



    /* Do not output the last pic after seeking. */

    if (s->last_picture_ptr || s->low_delay) {

        *data_size = sizeof(AVFrame);

    }



    return buf_size;

}
