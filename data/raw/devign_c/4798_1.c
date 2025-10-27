static int vobsub_read_seek(AVFormatContext *s, int stream_index,

                            int64_t min_ts, int64_t ts, int64_t max_ts, int flags)

{

    MpegDemuxContext *vobsub = s->priv_data;



    /* Rescale requested timestamps based on the first stream (timebase is the

     * same for all subtitles stream within a .idx/.sub). Rescaling is done just

     * like in avformat_seek_file(). */

    if (stream_index == -1 && s->nb_streams != 1) {

        int i, ret = 0;

        AVRational time_base = s->streams[0]->time_base;

        ts = av_rescale_q(ts, AV_TIME_BASE_Q, time_base);

        min_ts = av_rescale_rnd(min_ts, time_base.den,

                                time_base.num * (int64_t)AV_TIME_BASE,

                                AV_ROUND_UP   | AV_ROUND_PASS_MINMAX);

        max_ts = av_rescale_rnd(max_ts, time_base.den,

                                time_base.num * (int64_t)AV_TIME_BASE,

                                AV_ROUND_DOWN | AV_ROUND_PASS_MINMAX);

        for (i = 0; i < s->nb_streams; i++) {

            int r = ff_subtitles_queue_seek(&vobsub->q[i], s, stream_index,

                                            min_ts, ts, max_ts, flags);

            if (r < 0)

                ret = r;

        }

        return ret;

    }





    return ff_subtitles_queue_seek(&vobsub->q[stream_index], s, stream_index,

                                   min_ts, ts, max_ts, flags);

}