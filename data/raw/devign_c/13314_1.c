static int r3d_seek(AVFormatContext *s, int stream_index, int64_t sample_time, int flags)

{

    AVStream *st = s->streams[0]; // video stream

    R3DContext *r3d = s->priv_data;

    int frame_num;



    if (!st->codec->time_base.num || !st->time_base.den)

        return -1;



    frame_num = sample_time*st->codec->time_base.den/

        ((int64_t)st->codec->time_base.num*st->time_base.den);

    av_dlog(s, "seek frame num %d timestamp %"PRId64"\n",

            frame_num, sample_time);



    if (frame_num < r3d->video_offsets_count) {

        avio_seek(s->pb, r3d->video_offsets_count, SEEK_SET);

    } else {

        av_log(s, AV_LOG_ERROR, "could not seek to frame %d\n", frame_num);

        return -1;

    }



    return 0;

}
