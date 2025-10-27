static void write_packet(OutputFile *of, AVPacket *pkt, OutputStream *ost)

{

    AVFormatContext *s = of->ctx;

    AVStream *st = ost->st;

    int ret;



    if (!of->header_written) {

        AVPacket tmp_pkt;

        /* the muxer is not initialized yet, buffer the packet */

        if (!av_fifo_space(ost->muxing_queue)) {

            int new_size = FFMIN(2 * av_fifo_size(ost->muxing_queue),

                                 ost->max_muxing_queue_size);

            if (new_size <= av_fifo_size(ost->muxing_queue)) {

                av_log(NULL, AV_LOG_ERROR,

                       "Too many packets buffered for output stream %d:%d.\n",

                       ost->file_index, ost->st->index);

                exit_program(1);

            }

            ret = av_fifo_realloc2(ost->muxing_queue, new_size);

            if (ret < 0)

                exit_program(1);

        }

        av_packet_move_ref(&tmp_pkt, pkt);

        av_fifo_generic_write(ost->muxing_queue, &tmp_pkt, sizeof(tmp_pkt), NULL);

        return;

    }



    if ((st->codecpar->codec_type == AVMEDIA_TYPE_VIDEO && video_sync_method == VSYNC_DROP) ||

        (st->codecpar->codec_type == AVMEDIA_TYPE_AUDIO && audio_sync_method < 0))

        pkt->pts = pkt->dts = AV_NOPTS_VALUE;



    /*

     * Audio encoders may split the packets --  #frames in != #packets out.

     * But there is no reordering, so we can limit the number of output packets

     * by simply dropping them here.

     * Counting encoded video frames needs to be done separately because of

     * reordering, see do_video_out()

     */

    if (!(st->codecpar->codec_type == AVMEDIA_TYPE_VIDEO && ost->encoding_needed)) {

        if (ost->frame_number >= ost->max_frames) {

            av_packet_unref(pkt);

            return;

        }

        ost->frame_number++;

    }

    if (st->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {

        int i;

        uint8_t *sd = av_packet_get_side_data(pkt, AV_PKT_DATA_QUALITY_STATS,

                                              NULL);

        ost->quality = sd ? AV_RL32(sd) : -1;

        ost->pict_type = sd ? sd[4] : AV_PICTURE_TYPE_NONE;



        for (i = 0; i<FF_ARRAY_ELEMS(ost->error); i++) {

            if (sd && i < sd[5])

                ost->error[i] = AV_RL64(sd + 8 + 8*i);

            else

                ost->error[i] = -1;

        }



        if (ost->frame_rate.num && ost->is_cfr) {

            if (pkt->duration > 0)

                av_log(NULL, AV_LOG_WARNING, "Overriding packet duration by frame rate, this should not happen\n");

            pkt->duration = av_rescale_q(1, av_inv_q(ost->frame_rate),

                                         ost->st->time_base);

        }

    }



    if (!(s->oformat->flags & AVFMT_NOTIMESTAMPS)) {

        if (pkt->dts != AV_NOPTS_VALUE &&

            pkt->pts != AV_NOPTS_VALUE &&

            pkt->dts > pkt->pts) {

            av_log(s, AV_LOG_WARNING, "Invalid DTS: %"PRId64" PTS: %"PRId64" in output stream %d:%d, replacing by guess\n",

                   pkt->dts, pkt->pts,

                   ost->file_index, ost->st->index);

            pkt->pts =

            pkt->dts = pkt->pts + pkt->dts + ost->last_mux_dts + 1

                     - FFMIN3(pkt->pts, pkt->dts, ost->last_mux_dts + 1)

                     - FFMAX3(pkt->pts, pkt->dts, ost->last_mux_dts + 1);

        }

        if ((st->codecpar->codec_type == AVMEDIA_TYPE_AUDIO || st->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) &&

            pkt->dts != AV_NOPTS_VALUE &&

            !(st->codecpar->codec_id == AV_CODEC_ID_VP9 && ost->stream_copy) &&

            ost->last_mux_dts != AV_NOPTS_VALUE) {

            int64_t max = ost->last_mux_dts + !(s->oformat->flags & AVFMT_TS_NONSTRICT);

            if (pkt->dts < max) {

                int loglevel = max - pkt->dts > 2 || st->codecpar->codec_type == AVMEDIA_TYPE_VIDEO ? AV_LOG_WARNING : AV_LOG_DEBUG;

                av_log(s, loglevel, "Non-monotonous DTS in output stream "

                       "%d:%d; previous: %"PRId64", current: %"PRId64"; ",

                       ost->file_index, ost->st->index, ost->last_mux_dts, pkt->dts);

                if (exit_on_error) {

                    av_log(NULL, AV_LOG_FATAL, "aborting.\n");

                    exit_program(1);

                }

                av_log(s, loglevel, "changing to %"PRId64". This may result "

                       "in incorrect timestamps in the output file.\n",

                       max);

                if (pkt->pts >= pkt->dts)

                    pkt->pts = FFMAX(pkt->pts, max);

                pkt->dts = max;

            }

        }

    }

    ost->last_mux_dts = pkt->dts;



    ost->data_size += pkt->size;

    ost->packets_written++;



    pkt->stream_index = ost->index;



    if (debug_ts) {

        av_log(NULL, AV_LOG_INFO, "muxer <- type:%s "

                "pkt_pts:%s pkt_pts_time:%s pkt_dts:%s pkt_dts_time:%s size:%d\n",

                av_get_media_type_string(ost->enc_ctx->codec_type),

                av_ts2str(pkt->pts), av_ts2timestr(pkt->pts, &ost->st->time_base),

                av_ts2str(pkt->dts), av_ts2timestr(pkt->dts, &ost->st->time_base),

                pkt->size

              );

    }



    ret = av_interleaved_write_frame(s, pkt);

    if (ret < 0) {

        print_error("av_interleaved_write_frame()", ret);

        main_return_code = 1;

        close_all_output_streams(ost, MUXER_FINISHED | ENCODER_FINISHED, ENCODER_FINISHED);

    }

    av_packet_unref(pkt);

}
