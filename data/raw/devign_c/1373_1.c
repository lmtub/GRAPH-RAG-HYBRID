static int ipmovie_read_packet(AVFormatContext *s,

                               AVPacket *pkt)

{

    IPMVEContext *ipmovie = (IPMVEContext *)s->priv_data;

    ByteIOContext *pb = &s->pb;

    int ret;



    ret = process_ipmovie_chunk(ipmovie, pb, pkt);

    if (ret == CHUNK_BAD)

        ret = AVERROR_INVALIDDATA;

    else if (ret == CHUNK_EOF)

        ret = AVERROR_IO;

    else if (ret == CHUNK_NOMEM)

        ret = AVERROR_NOMEM;

    else

        ret = 0;



    return ret;

}
