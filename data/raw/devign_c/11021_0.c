static int lmlm4_read_packet(AVFormatContext *s, AVPacket *pkt) {

    AVIOContext *pb = s->pb;

    int ret;

    unsigned int frame_type, packet_size, padding, frame_size;



    avio_rb16(pb);                       /* channel number */

    frame_type  = avio_rb16(pb);

    packet_size = avio_rb32(pb);

    padding     = -packet_size & 511;

    frame_size  = packet_size - 8;



    if (frame_type > LMLM4_MPEG1L2 || frame_type == LMLM4_INVALID) {

        av_log(s, AV_LOG_ERROR, "invalid or unsupported frame_type\n");

        return AVERROR(EIO);

    }

    if (packet_size > LMLM4_MAX_PACKET_SIZE) {

        av_log(s, AV_LOG_ERROR, "packet size exceeds maximum\n");

        return AVERROR(EIO);

    }



    if ((ret = av_get_packet(pb, pkt, frame_size)) <= 0)

        return AVERROR(EIO);



    avio_skip(pb, padding);



    switch (frame_type) {

        case LMLM4_I_FRAME:

            pkt->flags = AV_PKT_FLAG_KEY;

        case LMLM4_P_FRAME:

        case LMLM4_B_FRAME:

            pkt->stream_index = 0;

            break;

        case LMLM4_MPEG1L2:

            pkt->stream_index = 1;

            break;

    }



    return ret;

}
