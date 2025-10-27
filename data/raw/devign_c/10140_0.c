int ff_MPV_encode_picture(AVCodecContext *avctx, AVPacket *pkt,

                          const AVFrame *pic_arg, int *got_packet)

{

    MpegEncContext *s = avctx->priv_data;

    int i, stuffing_count, ret;

    int context_count = s->slice_context_count;



    s->picture_in_gop_number++;



    if (load_input_picture(s, pic_arg) < 0)

        return -1;



    if (select_input_picture(s) < 0) {

        return -1;

    }



    /* output? */

    if (s->new_picture.f->data[0]) {

        if (!pkt->data &&

            (ret = ff_alloc_packet(pkt, s->mb_width*s->mb_height*MAX_MB_BYTES)) < 0)

            return ret;

        if (s->mb_info) {

            s->mb_info_ptr = av_packet_new_side_data(pkt,

                                 AV_PKT_DATA_H263_MB_INFO,

                                 s->mb_width*s->mb_height*12);

            s->prev_mb_info = s->last_mb_info = s->mb_info_size = 0;

        }



        for (i = 0; i < context_count; i++) {

            int start_y = s->thread_context[i]->start_mb_y;

            int   end_y = s->thread_context[i]->  end_mb_y;

            int h       = s->mb_height;

            uint8_t *start = pkt->data + (size_t)(((int64_t) pkt->size) * start_y / h);

            uint8_t *end   = pkt->data + (size_t)(((int64_t) pkt->size) *   end_y / h);



            init_put_bits(&s->thread_context[i]->pb, start, end - start);

        }



        s->pict_type = s->new_picture.f->pict_type;

        //emms_c();

        ret = frame_start(s);

        if (ret < 0)

            return ret;

vbv_retry:

        if (encode_picture(s, s->picture_number) < 0)

            return -1;



        avctx->header_bits = s->header_bits;

        avctx->mv_bits     = s->mv_bits;

        avctx->misc_bits   = s->misc_bits;

        avctx->i_tex_bits  = s->i_tex_bits;

        avctx->p_tex_bits  = s->p_tex_bits;

        avctx->i_count     = s->i_count;

        // FIXME f/b_count in avctx

        avctx->p_count     = s->mb_num - s->i_count - s->skip_count;

        avctx->skip_count  = s->skip_count;



        frame_end(s);



        if (CONFIG_MJPEG_ENCODER && s->out_format == FMT_MJPEG)

            ff_mjpeg_encode_picture_trailer(&s->pb, s->header_bits);



        if (avctx->rc_buffer_size) {

            RateControlContext *rcc = &s->rc_context;

            int max_size = rcc->buffer_index * avctx->rc_max_available_vbv_use;



            if (put_bits_count(&s->pb) > max_size &&

                s->lambda < s->avctx->lmax) {

                s->next_lambda = FFMAX(s->lambda + 1, s->lambda *

                                       (s->qscale + 1) / s->qscale);

                if (s->adaptive_quant) {

                    int i;

                    for (i = 0; i < s->mb_height * s->mb_stride; i++)

                        s->lambda_table[i] =

                            FFMAX(s->lambda_table[i] + 1,

                                  s->lambda_table[i] * (s->qscale + 1) /

                                  s->qscale);

                }

                s->mb_skipped = 0;        // done in frame_start()

                // done in encode_picture() so we must undo it

                if (s->pict_type == AV_PICTURE_TYPE_P) {

                    if (s->flipflop_rounding          ||

                        s->codec_id == AV_CODEC_ID_H263P ||

                        s->codec_id == AV_CODEC_ID_MPEG4)

                        s->no_rounding ^= 1;

                }

                if (s->pict_type != AV_PICTURE_TYPE_B) {

                    s->time_base       = s->last_time_base;

                    s->last_non_b_time = s->time - s->pp_time;

                }

                for (i = 0; i < context_count; i++) {

                    PutBitContext *pb = &s->thread_context[i]->pb;

                    init_put_bits(pb, pb->buf, pb->buf_end - pb->buf);

                }

                goto vbv_retry;

            }



            assert(s->avctx->rc_max_rate);

        }



        if (s->flags & CODEC_FLAG_PASS1)

            ff_write_pass1_stats(s);



        for (i = 0; i < 4; i++) {

            s->current_picture_ptr->f->error[i] = s->current_picture.f->error[i];

            avctx->error[i] += s->current_picture_ptr->f->error[i];

        }



        if (s->flags & CODEC_FLAG_PASS1)

            assert(avctx->header_bits + avctx->mv_bits + avctx->misc_bits +

                   avctx->i_tex_bits + avctx->p_tex_bits ==

                       put_bits_count(&s->pb));

        flush_put_bits(&s->pb);

        s->frame_bits  = put_bits_count(&s->pb);



        stuffing_count = ff_vbv_update(s, s->frame_bits);

        if (stuffing_count) {

            if (s->pb.buf_end - s->pb.buf - (put_bits_count(&s->pb) >> 3) <

                    stuffing_count + 50) {

                av_log(s->avctx, AV_LOG_ERROR, "stuffing too large\n");

                return -1;

            }



            switch (s->codec_id) {

            case AV_CODEC_ID_MPEG1VIDEO:

            case AV_CODEC_ID_MPEG2VIDEO:

                while (stuffing_count--) {

                    put_bits(&s->pb, 8, 0);

                }

            break;

            case AV_CODEC_ID_MPEG4:

                put_bits(&s->pb, 16, 0);

                put_bits(&s->pb, 16, 0x1C3);

                stuffing_count -= 4;

                while (stuffing_count--) {

                    put_bits(&s->pb, 8, 0xFF);

                }

            break;

            default:

                av_log(s->avctx, AV_LOG_ERROR, "vbv buffer overflow\n");

            }

            flush_put_bits(&s->pb);

            s->frame_bits  = put_bits_count(&s->pb);

        }



        /* update mpeg1/2 vbv_delay for CBR */

        if (s->avctx->rc_max_rate                          &&

            s->avctx->rc_min_rate == s->avctx->rc_max_rate &&

            s->out_format == FMT_MPEG1                     &&

            90000LL * (avctx->rc_buffer_size - 1) <=

                s->avctx->rc_max_rate * 0xFFFFLL) {

            int vbv_delay, min_delay;

            double inbits  = s->avctx->rc_max_rate *

                             av_q2d(s->avctx->time_base);

            int    minbits = s->frame_bits - 8 *

                             (s->vbv_delay_ptr - s->pb.buf - 1);

            double bits    = s->rc_context.buffer_index + minbits - inbits;



            if (bits < 0)

                av_log(s->avctx, AV_LOG_ERROR,

                       "Internal error, negative bits\n");



            assert(s->repeat_first_field == 0);



            vbv_delay = bits * 90000 / s->avctx->rc_max_rate;

            min_delay = (minbits * 90000LL + s->avctx->rc_max_rate - 1) /

                        s->avctx->rc_max_rate;



            vbv_delay = FFMAX(vbv_delay, min_delay);



            assert(vbv_delay < 0xFFFF);



            s->vbv_delay_ptr[0] &= 0xF8;

            s->vbv_delay_ptr[0] |= vbv_delay >> 13;

            s->vbv_delay_ptr[1]  = vbv_delay >> 5;

            s->vbv_delay_ptr[2] &= 0x07;

            s->vbv_delay_ptr[2] |= vbv_delay << 3;

            avctx->vbv_delay     = vbv_delay * 300;

        }

        s->total_bits     += s->frame_bits;

        avctx->frame_bits  = s->frame_bits;



        pkt->pts = s->current_picture.f->pts;

        if (!s->low_delay) {

            if (!s->current_picture.f->coded_picture_number)

                pkt->dts = pkt->pts - s->dts_delta;

            else

                pkt->dts = s->reordered_pts;

            s->reordered_pts = s->input_picture[0]->f->pts;

        } else

            pkt->dts = pkt->pts;

        if (s->current_picture.f->key_frame)

            pkt->flags |= AV_PKT_FLAG_KEY;

        if (s->mb_info)

            av_packet_shrink_side_data(pkt, AV_PKT_DATA_H263_MB_INFO, s->mb_info_size);

    } else {

        s->frame_bits = 0;

    }

    assert((s->frame_bits & 7) == 0);



    pkt->size = s->frame_bits / 8;

    *got_packet = !!pkt->size;

    return 0;

}
