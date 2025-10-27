static int gxf_packet(AVFormatContext *s, AVPacket *pkt) {

    ByteIOContext *pb = s->pb;

    pkt_type_t pkt_type;

    int pkt_len;

    while (!url_feof(pb)) {

        int track_type, track_id, ret;

        int field_nr;

        if (!parse_packet_header(pb, &pkt_type, &pkt_len)) {

            if (!url_feof(pb))

                av_log(s, AV_LOG_ERROR, "GXF: sync lost\n");

            return -1;

        }

        if (pkt_type == PKT_FLT) {

            gxf_read_index(s, pkt_len);

            continue;

        }

        if (pkt_type != PKT_MEDIA) {

            url_fskip(pb, pkt_len);

            continue;

        }

        if (pkt_len < 16) {

            av_log(s, AV_LOG_ERROR, "GXF: invalid media packet length\n");

            continue;

        }

        pkt_len -= 16;

        track_type = get_byte(pb);

        track_id = get_byte(pb);

        field_nr = get_be32(pb);

        get_be32(pb); // field information

        get_be32(pb); // "timeline" field number

        get_byte(pb); // flags

        get_byte(pb); // reserved

        // NOTE: there is also data length information in the

        // field information, it might be better to take this into account

        // as well.

        ret = av_get_packet(pb, pkt, pkt_len);

        pkt->stream_index = get_sindex(s, track_id, track_type);

        pkt->dts = field_nr;

        return ret;

    }

    return AVERROR(EIO);

}
