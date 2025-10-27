static int dv_read_seek(AVFormatContext *s, int stream_index,

                       int64_t timestamp, int flags)

{

    RawDVContext *r   = s->priv_data;

    DVDemuxContext *c = r->dv_demux;

    int64_t offset    = dv_frame_offset(s, c, timestamp, flags);



    dv_offset_reset(c, offset / c->sys->frame_size);



    offset = avio_seek(s->pb, offset, SEEK_SET);

    return (offset < 0) ? offset : 0;

}
