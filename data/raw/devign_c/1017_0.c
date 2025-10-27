static void estimate_timings_from_pts(AVFormatContext *ic, int64_t old_offset)

{

    AVPacket pkt1, *pkt = &pkt1;

    AVStream *st;

    int read_size, i, ret;

    int64_t end_time;

    int64_t filesize, offset, duration;

    int retry = 0;



    /* flush packet queue */

    flush_packet_queue(ic);



    for (i = 0; i < ic->nb_streams; i++) {

        st = ic->streams[i];

        if (st->start_time == AV_NOPTS_VALUE &&

            st->first_dts == AV_NOPTS_VALUE &&

            st->codec->codec_type != AVMEDIA_TYPE_UNKNOWN)

            av_log(st->codec, AV_LOG_WARNING,

                   "start time for stream %d is not set in estimate_timings_from_pts\n", i);



        if (st->parser) {

            av_parser_close(st->parser);

            st->parser = NULL;

        }

    }



    /* estimate the end time (duration) */

    /* XXX: may need to support wrapping */

    filesize = ic->pb ? avio_size(ic->pb) : 0;

    end_time = AV_NOPTS_VALUE;

    do {

        offset = filesize - (DURATION_MAX_READ_SIZE << retry);

        if (offset < 0)

            offset = 0;



        avio_seek(ic->pb, offset, SEEK_SET);

        read_size = 0;

        for (;;) {

            if (read_size >= DURATION_MAX_READ_SIZE << (FFMAX(retry - 1, 0)))

                break;



            do {

                ret = ff_read_packet(ic, pkt);

            } while (ret == AVERROR(EAGAIN));

            if (ret != 0)

                break;

            read_size += pkt->size;

            st         = ic->streams[pkt->stream_index];

            if (pkt->pts != AV_NOPTS_VALUE &&

                (st->start_time != AV_NOPTS_VALUE ||

                 st->first_dts  != AV_NOPTS_VALUE)) {

                duration = end_time = pkt->pts + pkt->duration;

                if (st->start_time != AV_NOPTS_VALUE)

                    duration -= st->start_time;

                else

                    duration -= st->first_dts;

                if (duration > 0) {

                    if (st->duration == AV_NOPTS_VALUE || st->info->last_duration<= 0 ||

                        (st->duration < duration && FFABS(duration - st->info->last_duration) < 60LL*st->time_base.den / st->time_base.num))

                        st->duration = duration;

                    st->info->last_duration = duration;

                }

            }

            av_free_packet(pkt);

        }

    } while (end_time == AV_NOPTS_VALUE &&

             filesize > (DURATION_MAX_READ_SIZE << retry) &&

             ++retry <= DURATION_MAX_RETRY);



    /* warn about audio/video streams which duration could not be estimated */

    for (i = 0; i < ic->nb_streams; i++) {

        st = ic->streams[i];

        if (st->duration == AV_NOPTS_VALUE) {

            switch (st->codec->codec_type) {

            case AVMEDIA_TYPE_VIDEO:

            case AVMEDIA_TYPE_AUDIO:

                if (st->start_time != AV_NOPTS_VALUE || st->first_dts  != AV_NOPTS_VALUE) {

                    av_log(ic, AV_LOG_DEBUG, "stream %d : no PTS found at end of file, duration not set\n", i);

                } else

                    av_log(ic, AV_LOG_DEBUG, "stream %d : no TS found at start of file, duration not set\n", i);

            }

        }

    }

    fill_all_stream_timings(ic);



    avio_seek(ic->pb, old_offset, SEEK_SET);

    for (i = 0; i < ic->nb_streams; i++) {

        int j;



        st              = ic->streams[i];

        st->cur_dts     = st->first_dts;

        st->last_IP_pts = AV_NOPTS_VALUE;

        st->last_dts_for_order_check = AV_NOPTS_VALUE;

        for (j = 0; j < MAX_REORDER_DELAY + 1; j++)

            st->pts_buffer[j] = AV_NOPTS_VALUE;

    }

}
