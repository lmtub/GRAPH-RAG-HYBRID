static void av_estimate_timings_from_pts(AVFormatContext *ic)

{

    AVPacket pkt1, *pkt = &pkt1;

    AVStream *st;

    int read_size, i, ret;

    int64_t end_time;

    int64_t filesize, offset, duration;



    /* free previous packet */

    if (ic->cur_st && ic->cur_st->parser)

        av_free_packet(&ic->cur_pkt);

    ic->cur_st = NULL;



    /* flush packet queue */

    flush_packet_queue(ic);



    for(i=0;i<ic->nb_streams;i++) {

        st = ic->streams[i];

        if (st->parser) {

            av_parser_close(st->parser);

            st->parser= NULL;

        }

    }



    /* we read the first packets to get the first PTS (not fully

       accurate, but it is enough now) */

    url_fseek(&ic->pb, 0, SEEK_SET);

    read_size = 0;

    for(;;) {

        if (read_size >= DURATION_MAX_READ_SIZE)

            break;

        /* if all info is available, we can stop */

        for(i = 0;i < ic->nb_streams; i++) {

            st = ic->streams[i];

            if (st->start_time == AV_NOPTS_VALUE)

                break;

        }

        if (i == ic->nb_streams)

            break;



        ret = av_read_packet(ic, pkt);

        if (ret != 0)

            break;

        read_size += pkt->size;

        st = ic->streams[pkt->stream_index];

        if (pkt->pts != AV_NOPTS_VALUE) {

            if (st->start_time == AV_NOPTS_VALUE)

                st->start_time = pkt->pts;

        }

        av_free_packet(pkt);

    }



    /* estimate the end time (duration) */

    /* XXX: may need to support wrapping */

    filesize = ic->file_size;

    offset = filesize - DURATION_MAX_READ_SIZE;

    if (offset < 0)

        offset = 0;



    url_fseek(&ic->pb, offset, SEEK_SET);

    read_size = 0;

    for(;;) {

        if (read_size >= DURATION_MAX_READ_SIZE)

            break;

        /* if all info is available, we can stop */

        for(i = 0;i < ic->nb_streams; i++) {

            st = ic->streams[i];

            if (st->duration == AV_NOPTS_VALUE)

                break;

        }

        if (i == ic->nb_streams)

            break;



        ret = av_read_packet(ic, pkt);

        if (ret != 0)

            break;

        read_size += pkt->size;

        st = ic->streams[pkt->stream_index];

        if (pkt->pts != AV_NOPTS_VALUE) {

            end_time = pkt->pts;

            duration = end_time - st->start_time;

            if (duration > 0) {

                if (st->duration == AV_NOPTS_VALUE ||

                    st->duration < duration)

                    st->duration = duration;

            }

        }

        av_free_packet(pkt);

    }



    fill_all_stream_timings(ic);



    url_fseek(&ic->pb, 0, SEEK_SET);

}
