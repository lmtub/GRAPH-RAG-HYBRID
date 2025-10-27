static int ast_read_packet(AVFormatContext *s, AVPacket *pkt)

{

    uint32_t type, size;

    int64_t pos;

    int ret;



    if (avio_feof(s->pb))

        return AVERROR_EOF;



    pos  = avio_tell(s->pb);

    type = avio_rl32(s->pb);

    size = avio_rb32(s->pb);

    if (size > INT_MAX / s->streams[0]->codecpar->channels)

        return AVERROR_INVALIDDATA;



    size *= s->streams[0]->codecpar->channels;

    if ((ret = avio_skip(s->pb, 24)) < 0) // padding

        return ret;



    if (type == MKTAG('B','L','C','K')) {

        ret = av_get_packet(s->pb, pkt, size);

        pkt->stream_index = 0;

        pkt->pos = pos;

    } else {

        av_log(s, AV_LOG_ERROR, "unknown chunk %x\n", type);

        avio_skip(s->pb, size);

        ret = AVERROR_INVALIDDATA;

    }



    return ret;

}
