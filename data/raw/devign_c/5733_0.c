static int rtsp_read_seek(AVFormatContext *s, int stream_index,

                          int64_t timestamp, int flags)

{

    RTSPState *rt = s->priv_data;



    rt->seek_timestamp = av_rescale_q(timestamp, s->streams[stream_index]->time_base, AV_TIME_BASE_Q);

    switch(rt->state) {

    default:

    case RTSP_STATE_IDLE:

        break;

    case RTSP_STATE_PLAYING:

        if (rtsp_read_pause(s) != 0)

            return -1;

        rt->state = RTSP_STATE_SEEKING;

        if (rtsp_read_play(s) != 0)

            return -1;

        break;

    case RTSP_STATE_PAUSED:

        rt->state = RTSP_STATE_IDLE;

        break;

    }

    return 0;

}
