static int read_seek(AVFormatContext *s, int stream_index,

                     int64_t ts, int flags)

{

    WtvContext *wtv = s->priv_data;

    AVIOContext *pb = wtv->pb;

    AVStream *st = s->streams[0];

    int64_t ts_relative;

    int i;



    if ((flags & AVSEEK_FLAG_FRAME) || (flags & AVSEEK_FLAG_BYTE))

        return AVERROR(ENOSYS);



    /* timestamp adjustment is required because wtv->pts values are absolute,

     * whereas AVIndexEntry->timestamp values are relative to epoch. */

    ts_relative = ts;

    if (wtv->epoch != AV_NOPTS_VALUE)

        ts_relative -= wtv->epoch;



    i = ff_index_search_timestamp(wtv->index_entries, wtv->nb_index_entries, ts_relative, flags);

    if (i < 0) {

        if (wtv->last_valid_pts == AV_NOPTS_VALUE || ts < wtv->last_valid_pts)

            avio_seek(pb, 0, SEEK_SET);

        else if (st->duration != AV_NOPTS_VALUE && ts_relative > st->duration && wtv->nb_index_entries)

            avio_seek(pb, wtv->index_entries[wtv->nb_index_entries - 1].pos, SEEK_SET);

        if (parse_chunks(s, SEEK_TO_PTS, ts, 0) < 0)

            return AVERROR(ERANGE);

        return 0;

    }

    wtv->pts = wtv->index_entries[i].timestamp;

    if (wtv->epoch != AV_NOPTS_VALUE)

        wtv->pts += wtv->epoch;

    wtv->last_valid_pts = wtv->pts;

    avio_seek(pb, wtv->index_entries[i].pos, SEEK_SET);

    return 0;

}
