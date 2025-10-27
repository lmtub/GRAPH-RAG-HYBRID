int ff_rv34_decode_frame(AVCodecContext *avctx,

                            void *data, int *data_size,

                            AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    RV34DecContext *r = avctx->priv_data;

    MpegEncContext *s = &r->s;

    AVFrame *pict = data;

    SliceInfo si;

    int i;

    int slice_count;

    const uint8_t *slices_hdr = NULL;

    int last = 0;



    /* no supplementary picture */

    if (buf_size == 0) {

        /* special case for last picture */

        if (s->low_delay==0 && s->next_picture_ptr) {

            *pict = *(AVFrame*)s->next_picture_ptr;

            s->next_picture_ptr = NULL;



            *data_size = sizeof(AVFrame);

        }

        return 0;

    }



    if(!avctx->slice_count){

        slice_count = (*buf++) + 1;

        slices_hdr = buf + 4;

        buf += 8 * slice_count;

    }else

        slice_count = avctx->slice_count;



    //parse first slice header to check whether this frame can be decoded

    if(get_slice_offset(avctx, slices_hdr, 0) > buf_size){

        av_log(avctx, AV_LOG_ERROR, "Slice offset is greater than frame size\n");

        return -1;

    }

    init_get_bits(&s->gb, buf+get_slice_offset(avctx, slices_hdr, 0), (buf_size-get_slice_offset(avctx, slices_hdr, 0))*8);

    if(r->parse_slice_header(r, &r->s.gb, &si) < 0 || si.start){

        av_log(avctx, AV_LOG_ERROR, "First slice header is incorrect\n");

        return -1;

    }

    if ((!s->last_picture_ptr || !s->last_picture_ptr->f.data[0]) && si.type == AV_PICTURE_TYPE_B)

        return -1;

    if(   (avctx->skip_frame >= AVDISCARD_NONREF && si.type==AV_PICTURE_TYPE_B)

       || (avctx->skip_frame >= AVDISCARD_NONKEY && si.type!=AV_PICTURE_TYPE_I)

       ||  avctx->skip_frame >= AVDISCARD_ALL)

        return buf_size;



    for(i = 0; i < slice_count; i++){

        int offset = get_slice_offset(avctx, slices_hdr, i);

        int size;

        if(i+1 == slice_count)

            size = buf_size - offset;

        else

            size = get_slice_offset(avctx, slices_hdr, i+1) - offset;



        if(offset > buf_size){

            av_log(avctx, AV_LOG_ERROR, "Slice offset is greater than frame size\n");

            break;

        }



        r->si.end = s->mb_width * s->mb_height;

        if(i+1 < slice_count){

            init_get_bits(&s->gb, buf+get_slice_offset(avctx, slices_hdr, i+1), (buf_size-get_slice_offset(avctx, slices_hdr, i+1))*8);

            if(r->parse_slice_header(r, &r->s.gb, &si) < 0){

                if(i+2 < slice_count)

                    size = get_slice_offset(avctx, slices_hdr, i+2) - offset;

                else

                    size = buf_size - offset;

            }else

                r->si.end = si.start;

        }

        last = rv34_decode_slice(r, r->si.end, buf + offset, size);

        s->mb_num_left = r->s.mb_x + r->s.mb_y*r->s.mb_width - r->si.start;

        if(last)

            break;

    }



    if(last && s->current_picture_ptr){

        if(r->loop_filter)

            r->loop_filter(r, s->mb_height - 1);

        ff_er_frame_end(s);

        MPV_frame_end(s);

        if (s->pict_type == AV_PICTURE_TYPE_B || s->low_delay) {

            *pict = *(AVFrame*)s->current_picture_ptr;

        } else if (s->last_picture_ptr != NULL) {

            *pict = *(AVFrame*)s->last_picture_ptr;

        }



        if(s->last_picture_ptr || s->low_delay){

            *data_size = sizeof(AVFrame);

            ff_print_debug_info(s, pict);

        }

        s->current_picture_ptr = NULL; //so we can detect if frame_end wasnt called (find some nicer solution...)

    }

    return buf_size;

}
