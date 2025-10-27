static int nsv_read_seek(AVFormatContext *s, int stream_index, int64_t timestamp, int flags)

{

    NSVContext *nsv = s->priv_data;

    AVStream *st = s->streams[stream_index];

    NSVStream *nst = st->priv_data;

    int index;



    index = av_index_search_timestamp(st, timestamp, flags);

    if(index < 0)

        return -1;



    avio_seek(s->pb, st->index_entries[index].pos, SEEK_SET);

    nst->frame_offset = st->index_entries[index].timestamp;

    nsv->state = NSV_UNSYNC;

    return 0;

}
