static int read_seek(AVFormatContext *s, int stream_index, int64_t timestamp, int flags)

{

    AVStream *st = s->streams[stream_index];

    avio_seek(s->pb, FFMAX(timestamp, 0) * st->codec->width * st->codec->height * 4, SEEK_SET);

    return 0;

}
