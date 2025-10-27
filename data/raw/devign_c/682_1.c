int av_seek_frame(AVFormatContext *s, int stream_index, int64_t timestamp, int flags)

{

    int ret;

    AVStream *st;



    ff_read_frame_flush(s);



    if(flags & AVSEEK_FLAG_BYTE)

        return av_seek_frame_byte(s, stream_index, timestamp, flags);



    if(stream_index < 0){

        stream_index= av_find_default_stream_index(s);

        if(stream_index < 0)

            return -1;



        st= s->streams[stream_index];

       /* timestamp for default must be expressed in AV_TIME_BASE units */

        timestamp = av_rescale(timestamp, st->time_base.den, AV_TIME_BASE * (int64_t)st->time_base.num);

    }



    /* first, we try the format specific seek */

    if (s->iformat->read_seek)

        ret = s->iformat->read_seek(s, stream_index, timestamp, flags);

    else

        ret = -1;

    if (ret >= 0) {

        return 0;

    }



    if(s->iformat->read_timestamp)

        return av_seek_frame_binary(s, stream_index, timestamp, flags);

    else

        return av_seek_frame_generic(s, stream_index, timestamp, flags);

}
