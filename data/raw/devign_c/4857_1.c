static int yuv4_read_packet(AVFormatContext *s, AVPacket *pkt)

{

    int i;

    char header[MAX_FRAME_HEADER+1];

    int ret;

    int64_t off = avio_tell(s->pb);



    for (i = 0; i < MAX_FRAME_HEADER; i++) {

        header[i] = avio_r8(s->pb);

        if (header[i] == '\n') {

            header[i + 1] = 0;

            break;

        }

    }

    if (s->pb->error)

        return s->pb->error;

    else if (s->pb->eof_reached)

        return AVERROR_EOF;

    else if (i == MAX_FRAME_HEADER)

        return AVERROR_INVALIDDATA;



    if (strncmp(header, Y4M_FRAME_MAGIC, strlen(Y4M_FRAME_MAGIC)))

        return AVERROR_INVALIDDATA;



    ret = av_get_packet(s->pb, pkt, s->packet_size - Y4M_FRAME_MAGIC_LEN);

    if (ret < 0)

        return ret;

    else if (ret != s->packet_size - Y4M_FRAME_MAGIC_LEN)

        return s->pb->eof_reached ? AVERROR_EOF : AVERROR(EIO);



    pkt->stream_index = 0;

    pkt->pts = (off - s->internal->data_offset) / s->packet_size;

    pkt->duration = 1;

    return 0;

}
