static int rtp_parse_packet_internal(RTPDemuxContext *s, AVPacket *pkt,

                                     const uint8_t *buf, int len)

{

    unsigned int ssrc, h;

    int payload_type, seq, ret, flags = 0;

    int ext;

    AVStream *st;

    uint32_t timestamp;

    int rv = 0;



    ext          = buf[0] & 0x10;

    payload_type = buf[1] & 0x7f;

    if (buf[1] & 0x80)

        flags |= RTP_FLAG_MARKER;

    seq       = AV_RB16(buf + 2);

    timestamp = AV_RB32(buf + 4);

    ssrc      = AV_RB32(buf + 8);

    /* store the ssrc in the RTPDemuxContext */

    s->ssrc = ssrc;



    /* NOTE: we can handle only one payload type */

    if (s->payload_type != payload_type)

        return -1;



    st = s->st;

    // only do something with this if all the rtp checks pass...

    if (!rtp_valid_packet_in_sequence(&s->statistics, seq)) {

        av_log(st ? st->codec : NULL, AV_LOG_ERROR,

               "RTP: PT=%02x: bad cseq %04x expected=%04x\n",

               payload_type, seq, ((s->seq + 1) & 0xffff));

        return -1;

    }



    if (buf[0] & 0x20) {

        int padding = buf[len - 1];

        if (len >= 12 + padding)

            len -= padding;

    }



    s->seq = seq;

    len   -= 12;

    buf   += 12;



    /* RFC 3550 Section 5.3.1 RTP Header Extension handling */

    if (ext) {

        if (len < 4)

            return -1;

        /* calculate the header extension length (stored as number

         * of 32-bit words) */

        ext = (AV_RB16(buf + 2) + 1) << 2;



        if (len < ext)

            return -1;

        // skip past RTP header extension

        len -= ext;

        buf += ext;

    }



    if (!st) {

        /* specific MPEG2-TS demux support */

        ret = ff_mpegts_parse_packet(s->ts, pkt, buf, len);

        /* The only error that can be returned from ff_mpegts_parse_packet

         * is "no more data to return from the provided buffer", so return

         * AVERROR(EAGAIN) for all errors */

        if (ret < 0)

            return AVERROR(EAGAIN);

        if (ret < len) {

            s->read_buf_size = len - ret;

            memcpy(s->buf, buf + ret, s->read_buf_size);

            s->read_buf_index = 0;

            return 1;

        }

        return 0;

    } else if (s->parse_packet) {

        rv = s->parse_packet(s->ic, s->dynamic_protocol_context,

                             s->st, pkt, &timestamp, buf, len, flags);

    } else {

        /* At this point, the RTP header has been stripped;

         * This is ASSUMING that there is only 1 CSRC, which isn't wise. */

        switch (st->codec->codec_id) {

        case AV_CODEC_ID_MP2:

        case AV_CODEC_ID_MP3:

            /* better than nothing: skip MPEG audio RTP header */

            if (len <= 4)

                return -1;

            h    = AV_RB32(buf);

            len -= 4;

            buf += 4;

            av_new_packet(pkt, len);

            memcpy(pkt->data, buf, len);

            break;

        case AV_CODEC_ID_MPEG1VIDEO:

        case AV_CODEC_ID_MPEG2VIDEO:

            /* better than nothing: skip MPEG video RTP header */

            if (len <= 4)

                return -1;

            h    = AV_RB32(buf);

            buf += 4;

            len -= 4;

            if (h & (1 << 26)) {

                /* MPEG-2 */

                if (len <= 4)

                    return -1;

                buf += 4;

                len -= 4;

            }

            av_new_packet(pkt, len);

            memcpy(pkt->data, buf, len);

            break;

        default:

            av_new_packet(pkt, len);

            memcpy(pkt->data, buf, len);

            break;

        }



        pkt->stream_index = st->index;

    }



    // now perform timestamp things....

    finalize_packet(s, pkt, timestamp);



    return rv;

}
