static int flv_read_header(AVFormatContext *s)

{

    int flags;

    FLVContext *flv = s->priv_data;

    int offset;



    avio_skip(s->pb, 4);

    flags = avio_r8(s->pb);



    flv->missing_streams = flags & (FLV_HEADER_FLAG_HASVIDEO | FLV_HEADER_FLAG_HASAUDIO);



    s->ctx_flags |= AVFMTCTX_NOHEADER;



    offset = avio_rb32(s->pb);

    avio_seek(s->pb, offset, SEEK_SET);

    avio_skip(s->pb, 4);



    s->start_time = 0;

    flv->sum_flv_tag_size = 0;

    flv->last_keyframe_stream_index = -1;



    return 0;

}
