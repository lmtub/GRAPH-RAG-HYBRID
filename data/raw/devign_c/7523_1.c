int ff_h263_decode_frame(AVCodecContext *avctx, void *data, int *got_frame,

                         AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size       = avpkt->size;

    MpegEncContext *s  = avctx->priv_data;

    int ret;

    AVFrame *pict = data;



    s->flags  = avctx->flags;

    s->flags2 = avctx->flags2;



    /* no supplementary picture */

    if (buf_size == 0) {

        /* special case for last picture */

        if (s->low_delay == 0 && s->next_picture_ptr) {

            if ((ret = av_frame_ref(pict, &s->next_picture_ptr->f)) < 0)

                return ret;

            s->next_picture_ptr = NULL;



            *got_frame = 1;

        }



        return 0;

    }



    if (s->flags & CODEC_FLAG_TRUNCATED) {

        int next;



        if (CONFIG_MPEG4_DECODER && s->codec_id == AV_CODEC_ID_MPEG4) {

            next = ff_mpeg4_find_frame_end(&s->parse_context, buf, buf_size);

        } else if (CONFIG_H263_DECODER && s->codec_id == AV_CODEC_ID_H263) {

            next = ff_h263_find_frame_end(&s->parse_context, buf, buf_size);

        } else {

            av_log(s->avctx, AV_LOG_ERROR,

                   "this codec does not support truncated bitstreams\n");

            return AVERROR(ENOSYS);

        }



        if (ff_combine_frame(&s->parse_context, next, (const uint8_t **)&buf,

                             &buf_size) < 0)

            return buf_size;

    }



    if (s->bitstream_buffer_size && (s->divx_packed || buf_size < 20)) // divx 5.01+/xvid frame reorder

        ret = init_get_bits8(&s->gb, s->bitstream_buffer,

                             s->bitstream_buffer_size);

    else

        ret = init_get_bits8(&s->gb, buf, buf_size);

    s->bitstream_buffer_size = 0;



    if (ret < 0)

        return ret;



    if (!s->context_initialized)

        // we need the idct permutaton for reading a custom matrix

        if ((ret = ff_MPV_common_init(s)) < 0)

            return ret;



    /* We need to set current_picture_ptr before reading the header,

     * otherwise we cannot store anyting in there */

    if (s->current_picture_ptr == NULL || s->current_picture_ptr->f.data[0]) {

        int i = ff_find_unused_picture(s, 0);

        if (i < 0)

            return i;

        s->current_picture_ptr = &s->picture[i];

    }



    /* let's go :-) */

    if (CONFIG_WMV2_DECODER && s->msmpeg4_version == 5) {

        ret = ff_wmv2_decode_picture_header(s);

    } else if (CONFIG_MSMPEG4_DECODER && s->msmpeg4_version) {

        ret = ff_msmpeg4_decode_picture_header(s);

    } else if (CONFIG_MPEG4_DECODER && avctx->codec_id == AV_CODEC_ID_MPEG4) {

        if (s->avctx->extradata_size && s->picture_number == 0) {

            GetBitContext gb;



            ret = init_get_bits8(&gb, s->avctx->extradata,

                                 s->avctx->extradata_size);

            if (ret < 0)

                return ret;

            ff_mpeg4_decode_picture_header(avctx->priv_data, &gb);

        }

        ret = ff_mpeg4_decode_picture_header(avctx->priv_data, &s->gb);

    } else if (CONFIG_H263I_DECODER && s->codec_id == AV_CODEC_ID_H263I) {

        ret = ff_intel_h263_decode_picture_header(s);

    } else if (CONFIG_FLV_DECODER && s->h263_flv) {

        ret = ff_flv_decode_picture_header(s);

    } else {

        ret = ff_h263_decode_picture_header(s);

    }



    if (ret == FRAME_SKIPPED)

        return get_consumed_bytes(s, buf_size);



    /* skip if the header was thrashed */

    if (ret < 0) {

        av_log(s->avctx, AV_LOG_ERROR, "header damaged\n");

        return ret;

    }



    avctx->has_b_frames = !s->low_delay;



#define SET_QPEL_FUNC(postfix1, postfix2)                           \

    s->dsp.put_        ## postfix1 = ff_put_        ## postfix2;    \

    s->dsp.put_no_rnd_ ## postfix1 = ff_put_no_rnd_ ## postfix2;    \

    s->dsp.avg_        ## postfix1 = ff_avg_        ## postfix2;



    if (s->workaround_bugs & FF_BUG_STD_QPEL) {

        SET_QPEL_FUNC(qpel_pixels_tab[0][5], qpel16_mc11_old_c)

        SET_QPEL_FUNC(qpel_pixels_tab[0][7], qpel16_mc31_old_c)

        SET_QPEL_FUNC(qpel_pixels_tab[0][9], qpel16_mc12_old_c)

        SET_QPEL_FUNC(qpel_pixels_tab[0][11], qpel16_mc32_old_c)

        SET_QPEL_FUNC(qpel_pixels_tab[0][13], qpel16_mc13_old_c)

        SET_QPEL_FUNC(qpel_pixels_tab[0][15], qpel16_mc33_old_c)



        SET_QPEL_FUNC(qpel_pixels_tab[1][5], qpel8_mc11_old_c)

        SET_QPEL_FUNC(qpel_pixels_tab[1][7], qpel8_mc31_old_c)

        SET_QPEL_FUNC(qpel_pixels_tab[1][9], qpel8_mc12_old_c)

        SET_QPEL_FUNC(qpel_pixels_tab[1][11], qpel8_mc32_old_c)

        SET_QPEL_FUNC(qpel_pixels_tab[1][13], qpel8_mc13_old_c)

        SET_QPEL_FUNC(qpel_pixels_tab[1][15], qpel8_mc33_old_c)

    }



    /* After H263 & mpeg4 header decode we have the height, width,

     * and other parameters. So then we could init the picture.

     * FIXME: By the way H263 decoder is evolving it should have

     * an H263EncContext */

    if (s->width  != avctx->coded_width  ||

        s->height != avctx->coded_height ||

        s->context_reinit) {

        /* H.263 could change picture size any time */

        s->context_reinit = 0;



        ret = ff_set_dimensions(avctx, s->width, s->height);

        if (ret < 0)

            return ret;



        if ((ret = ff_MPV_common_frame_size_change(s)))

            return ret;

    }



    if (s->codec_id == AV_CODEC_ID_H263  ||

        s->codec_id == AV_CODEC_ID_H263P ||

        s->codec_id == AV_CODEC_ID_H263I)

        s->gob_index = ff_h263_get_gob_height(s);



    // for skipping the frame

    s->current_picture.f.pict_type = s->pict_type;

    s->current_picture.f.key_frame = s->pict_type == AV_PICTURE_TYPE_I;



    /* skip B-frames if we don't have reference frames */

    if (s->last_picture_ptr == NULL &&

        (s->pict_type == AV_PICTURE_TYPE_B || s->droppable))

        return get_consumed_bytes(s, buf_size);

    if ((avctx->skip_frame >= AVDISCARD_NONREF &&

         s->pict_type == AV_PICTURE_TYPE_B)    ||

        (avctx->skip_frame >= AVDISCARD_NONKEY &&

         s->pict_type != AV_PICTURE_TYPE_I)    ||

        avctx->skip_frame >= AVDISCARD_ALL)

        return get_consumed_bytes(s, buf_size);



    if (s->next_p_frame_damaged) {

        if (s->pict_type == AV_PICTURE_TYPE_B)

            return get_consumed_bytes(s, buf_size);

        else

            s->next_p_frame_damaged = 0;

    }



    if ((!s->no_rounding) || s->pict_type == AV_PICTURE_TYPE_B) {

        s->me.qpel_put = s->dsp.put_qpel_pixels_tab;

        s->me.qpel_avg = s->dsp.avg_qpel_pixels_tab;

    } else {

        s->me.qpel_put = s->dsp.put_no_rnd_qpel_pixels_tab;

        s->me.qpel_avg = s->dsp.avg_qpel_pixels_tab;

    }



    if ((ret = ff_MPV_frame_start(s, avctx)) < 0)

        return ret;



    if (!s->divx_packed && !avctx->hwaccel)

        ff_thread_finish_setup(avctx);



    if (avctx->hwaccel) {

        ret = avctx->hwaccel->start_frame(avctx, s->gb.buffer,

                                          s->gb.buffer_end - s->gb.buffer);

        if (ret < 0 )

            return ret;

    }



    ff_mpeg_er_frame_start(s);



    /* the second part of the wmv2 header contains the MB skip bits which

     * are stored in current_picture->mb_type which is not available before

     * ff_MPV_frame_start() */

    if (CONFIG_WMV2_DECODER && s->msmpeg4_version == 5) {

        ret = ff_wmv2_decode_secondary_picture_header(s);

        if (ret < 0)

            return ret;

        if (ret == 1)

            goto intrax8_decoded;

    }



    /* decode each macroblock */

    s->mb_x = 0;

    s->mb_y = 0;



    ret = decode_slice(s);

    while (s->mb_y < s->mb_height) {

        if (s->msmpeg4_version) {

            if (s->slice_height == 0 || s->mb_x != 0 ||

                (s->mb_y % s->slice_height) != 0 || get_bits_left(&s->gb) < 0)

                break;

        } else {

            int prev_x = s->mb_x, prev_y = s->mb_y;

            if (ff_h263_resync(s) < 0)

                break;

            if (prev_y * s->mb_width + prev_x < s->mb_y * s->mb_width + s->mb_x)

                s->er.error_occurred = 1;

        }



        if (s->msmpeg4_version < 4 && s->h263_pred)

            ff_mpeg4_clean_buffers(s);



        if (decode_slice(s) < 0)

            ret = AVERROR_INVALIDDATA;

    }



    if (s->msmpeg4_version && s->msmpeg4_version < 4 &&

        s->pict_type == AV_PICTURE_TYPE_I)

        if (!CONFIG_MSMPEG4_DECODER ||

            ff_msmpeg4_decode_ext_header(s, buf_size) < 0)

            s->er.error_status_table[s->mb_num - 1] = ER_MB_ERROR;



    assert(s->bitstream_buffer_size == 0);



    if (CONFIG_MPEG4_DECODER && avctx->codec_id == AV_CODEC_ID_MPEG4)

        ff_mpeg4_frame_end(avctx, buf, buf_size);



intrax8_decoded:

    ff_er_frame_end(&s->er);



    if (avctx->hwaccel) {

        ret = avctx->hwaccel->end_frame(avctx);

        if (ret < 0)

            return ret;

    }



    ff_MPV_frame_end(s);



    if (!s->divx_packed && avctx->hwaccel)

        ff_thread_finish_setup(avctx);



    assert(s->current_picture.f.pict_type ==

           s->current_picture_ptr->f.pict_type);

    assert(s->current_picture.f.pict_type == s->pict_type);

    if (s->pict_type == AV_PICTURE_TYPE_B || s->low_delay) {

        if ((ret = av_frame_ref(pict, &s->current_picture_ptr->f)) < 0)

            return ret;

        ff_print_debug_info(s, s->current_picture_ptr);

    } else if (s->last_picture_ptr != NULL) {

        if ((ret = av_frame_ref(pict, &s->last_picture_ptr->f)) < 0)

            return ret;

        ff_print_debug_info(s, s->last_picture_ptr);

    }



    if (s->last_picture_ptr || s->low_delay)

        *got_frame = 1;



    if (ret && (avctx->err_recognition & AV_EF_EXPLODE))

        return ret;

    else

        return get_consumed_bytes(s, buf_size);

}
