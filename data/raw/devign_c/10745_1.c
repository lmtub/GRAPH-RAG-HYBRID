static int tta_read_seek(AVFormatContext *s, int stream_index, int64_t timestamp, int flags)

{

    TTAContext *c = s->priv_data;

    AVStream *st = s->streams[stream_index];

    int index = av_index_search_timestamp(st, timestamp, flags);

    if (index < 0)

        return -1;



    c->currentframe = index;

    avio_seek(s->pb, st->index_entries[index].pos, SEEK_SET);



    return 0;

}
