void avformat_close_input(AVFormatContext **ps)

{

    AVFormatContext *s = *ps;

    AVIOContext *pb = (s->iformat->flags & AVFMT_NOFILE) || (s->flags & AVFMT_FLAG_CUSTOM_IO) ?

                       NULL : s->pb;

    flush_packet_queue(s);

    if (s->iformat->read_close)

        s->iformat->read_close(s);

    avformat_free_context(s);

    *ps = NULL;

    if (pb)

        avio_close(pb);

}
