static int rv10_decode_frame(AVCodecContext *avctx, void *data, int *got_frame,

                             AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size       = avpkt->size;

    MpegEncContext *s = avctx->priv_data;

    AVFrame *pict = data;

    int i, ret;

    int slice_count;

    const uint8_t *slices_hdr = NULL;



    av_dlog(avctx, "*****frame %d size=%d\n", avctx->frame_number, buf_size);



    /* no supplementary picture */

    if (buf_size == 0) {

        return 0;

    }



    if (!avctx->slice_count) {

        slice_count = (*buf++) + 1;

        buf_size--;



        if (!slice_count || buf_size <= 8 * slice_count) {

            av_log(avctx, AV_LOG_ERROR, "Invalid slice count: %d.\n",

                   slice_count);

            return AVERROR_INVALIDDATA;

        }



        slices_hdr = buf + 4;

        buf       += 8 * slice_count;

        buf_size  -= 8 * slice_count;

    } else

        slice_count = avctx->slice_count;



    for (i = 0; i < slice_count; i++) {

        unsigned offset = get_slice_offset(avctx, slices_hdr, i);

        int size, size2;



        if (offset >= buf_size)

            return AVERROR_INVALIDDATA;



        if (i + 1 == slice_count)

            size = buf_size - offset;

        else

            size = get_slice_offset(avctx, slices_hdr, i + 1) - offset;



        if (i + 2 >= slice_count)

            size2 = buf_size - offset;

        else

            size2 = get_slice_offset(avctx, slices_hdr, i + 2) - offset;



        if (size <= 0 || size2 <= 0 ||

            offset + FFMAX(size, size2) > buf_size)

            return AVERROR_INVALIDDATA;



        if ((ret = rv10_decode_packet(avctx, buf + offset, size, size2)) < 0)

            return ret;



        if (ret > 8 * size)

            i++;

    }



    if (s->current_picture_ptr != NULL && s->mb_y >= s->mb_height) {

        ff_er_frame_end(&s->er);

        ff_mpv_frame_end(s);



        if (s->pict_type == AV_PICTURE_TYPE_B || s->low_delay) {

            if ((ret = av_frame_ref(pict, s->current_picture_ptr->f)) < 0)

                return ret;

            ff_print_debug_info(s, s->current_picture_ptr);

        } else if (s->last_picture_ptr != NULL) {

            if ((ret = av_frame_ref(pict, s->last_picture_ptr->f)) < 0)

                return ret;

            ff_print_debug_info(s, s->last_picture_ptr);

        }



        if (s->last_picture_ptr || s->low_delay) {

            *got_frame = 1;

        }



        // so we can detect if frame_end was not called (find some nicer solution...)

        s->current_picture_ptr = NULL;

    }



    return avpkt->size;

}
