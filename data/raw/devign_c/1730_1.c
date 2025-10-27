static int mpc8_read_seek(AVFormatContext *s, int stream_index, int64_t timestamp, int flags)

{

    AVStream *st = s->streams[stream_index];

    MPCContext *c = s->priv_data;

    int index = av_index_search_timestamp(st, timestamp, flags);



    if(index < 0) return -1;

    avio_seek(s->pb, st->index_entries[index].pos, SEEK_SET);

    c->frame = st->index_entries[index].timestamp;

    return 0;

}
