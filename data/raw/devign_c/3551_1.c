static int tmv_read_seek(AVFormatContext *s, int stream_index,

                         int64_t timestamp, int flags)

{

    TMVContext *tmv = s->priv_data;

    int64_t pos;



    if (stream_index)

        return -1;



    pos = timestamp *

          (tmv->audio_chunk_size + tmv->video_chunk_size + tmv->padding);



    avio_seek(s->pb, pos + TMV_HEADER_SIZE, SEEK_SET);

    tmv->stream_index = 0;

    return 0;

}
