static void write_packet(AVFormatContext *s, AVPacket *pkt, OutputStream *ost)

{

    AVStream *st = ost->st;

    int ret;



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

        uint8_t *sd = av_packet_get_side_data(pkt, AV_PKT_DATA_QUALITY_FACTOR,

                                              NULL);

        ost->quality = sd ? *(int *)sd : -1;



        if (ost->frame_rate.num) {

            pkt->duration = av_rescale_q(1, av_inv_q(ost->frame_rate),

                                         ost->st->time_base);

        }

    }



    if (!(s->oformat->flags & AVFMT_NOTIMESTAMPS) &&

        ost->last_mux_dts != AV_NOPTS_VALUE &&

        pkt->dts < ost->last_mux_dts + !(s->oformat->flags & AVFMT_TS_NONSTRICT)) {

        av_log(NULL, AV_LOG_WARNING, "Non-monotonous DTS in output stream "

               "%d:%d; previous: %"PRId64", current: %"PRId64"; ",

               ost->file_index, ost->st->index, ost->last_mux_dts, pkt->dts);

        if (exit_on_error) {

            av_log(NULL, AV_LOG_FATAL, "aborting.\n");

            exit_program(1);

        }

        av_log(NULL, AV_LOG_WARNING, "changing to %"PRId64". This may result "

               "in incorrect timestamps in the output file.\n",

               ost->last_mux_dts + 1);

        pkt->dts = ost->last_mux_dts + 1;

        if (pkt->pts != AV_NOPTS_VALUE)

            pkt->pts = FFMAX(pkt->pts, pkt->dts);

    }

    ost->last_mux_dts = pkt->dts;



    ost->data_size += pkt->size;

    ost->packets_written++;



    pkt->stream_index = ost->index;

    ret = av_interleaved_write_frame(s, pkt);

    if (ret < 0) {

        print_error("av_interleaved_write_frame()", ret);

        exit_program(1);

    }

}
