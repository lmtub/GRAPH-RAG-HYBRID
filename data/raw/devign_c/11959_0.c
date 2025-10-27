static void av_estimate_timings_from_pts(AVFormatContext *ic)

{

    AVPacket pkt1, *pkt = &pkt1;

    AVStream *st;

    int read_size, i, ret;

    int64_t start_time, end_time, end_time1;

    int64_t filesize, offset, duration;

    

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

                st->start_time = (int64_t)((double)pkt->pts * ic->pts_num * (double)AV_TIME_BASE / ic->pts_den);

         }

         av_free_packet(pkt);

     }



    /* we compute the minimum start_time and use it as default */

    start_time = MAXINT64;

    for(i = 0; i < ic->nb_streams; i++) {

        st = ic->streams[i];

        if (st->start_time != AV_NOPTS_VALUE &&

            st->start_time < start_time)

            start_time = st->start_time;

    }

    fprintf(stderr, "start=%lld\n", start_time);

    if (start_time != MAXINT64)

        ic->start_time = start_time;

    

    /* estimate the end time (duration) */

    /* XXX: may need to support wrapping */

    filesize = ic->file_size;

    offset = filesize - DURATION_MAX_READ_SIZE;

    if (offset < 0)

        offset = 0;



    /* flush packet queue */

    flush_packet_queue(ic);



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

            end_time = (int64_t)((double)pkt->pts * ic->pts_num * (double)AV_TIME_BASE / ic->pts_den);

            duration = end_time - st->start_time;

            if (duration > 0) {

                if (st->duration == AV_NOPTS_VALUE ||

                    st->duration < duration)

                    st->duration = duration;

            }

        }

        av_free_packet(pkt);

    }

    

    /* estimate total duration */

    end_time = MININT64;

    for(i = 0;i < ic->nb_streams; i++) {

        st = ic->streams[i];

        if (st->duration != AV_NOPTS_VALUE) {

            end_time1 = st->start_time + st->duration;

            if (end_time1 > end_time)

                end_time = end_time1;

        }

    }

    

    /* update start_time (new stream may have been created, so we do

       it at the end */

    if (ic->start_time != AV_NOPTS_VALUE) {

        for(i = 0; i < ic->nb_streams; i++) {

            st = ic->streams[i];

            if (st->start_time == AV_NOPTS_VALUE)

                st->start_time = ic->start_time;

        }

    }



    if (end_time != MININT64) {

        /* put dummy values for duration if needed */

        for(i = 0;i < ic->nb_streams; i++) {

            st = ic->streams[i];

            if (st->duration == AV_NOPTS_VALUE && 

                st->start_time != AV_NOPTS_VALUE)

                st->duration = end_time - st->start_time;

        }

        ic->duration = end_time - ic->start_time;

    }



    url_fseek(&ic->pb, 0, SEEK_SET);

}
