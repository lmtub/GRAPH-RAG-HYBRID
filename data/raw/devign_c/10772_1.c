static int sox_read_packet(AVFormatContext *s,

                           AVPacket *pkt)

{

    int ret, size;



    if (url_feof(s->pb))

        return AVERROR_EOF;



    size = SOX_SAMPLES*s->streams[0]->codec->block_align;

    ret = av_get_packet(s->pb, pkt, size);

    if (ret < 0)

        return AVERROR(EIO);


    pkt->stream_index = 0;

    pkt->size = ret;



    return 0;

}