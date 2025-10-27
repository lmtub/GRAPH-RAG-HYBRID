static int rtp_parse_one_packet(RTPDemuxContext *s, AVPacket *pkt,

                                uint8_t **bufptr, int len)

{

    uint8_t *buf = bufptr ? *bufptr : NULL;

    int flags = 0;

    uint32_t timestamp;

    int rv = 0;



    if (!buf) {

        /* If parsing of the previous packet actually returned 0 or an error,

         * there's nothing more to be parsed from that packet, but we may have

         * indicated that we can return the next enqueued packet. */

        if (s->prev_ret <= 0)

            return rtp_parse_queued_packet(s, pkt);

        /* return the next packets, if any */

        if (s->handler && s->handler->parse_packet) {

            /* timestamp should be overwritten by parse_packet, if not,

             * the packet is left with pts == AV_NOPTS_VALUE */

            timestamp = RTP_NOTS_VALUE;

            rv        = s->handler->parse_packet(s->ic, s->dynamic_protocol_context,

                                                 s->st, pkt, &timestamp, NULL, 0, 0,

                                                 flags);

            finalize_packet(s, pkt, timestamp);

            return rv;

        }

    }



    if (len < 12)

        return -1;



    if ((buf[0] & 0xc0) != (RTP_VERSION << 6))

        return -1;

    if (RTP_PT_IS_RTCP(buf[1])) {

        return rtcp_parse_packet(s, buf, len);

    }



    if (s->st) {

        int64_t received = av_gettime_relative();

        uint32_t arrival_ts = av_rescale_q(received, AV_TIME_BASE_Q,

                                           s->st->time_base);

        timestamp = AV_RB32(buf + 4);

        // Calculate the jitter immediately, before queueing the packet

        // into the reordering queue.

        rtcp_update_jitter(&s->statistics, timestamp, arrival_ts);

    }



    if ((s->seq == 0 && !s->queue) || s->queue_size <= 1) {

        /* First packet, or no reordering */

        return rtp_parse_packet_internal(s, pkt, buf, len);

    } else {

        uint16_t seq = AV_RB16(buf + 2);

        int16_t diff = seq - s->seq;

        if (diff < 0) {

            /* Packet older than the previously emitted one, drop */

            av_log(s->st ? s->st->codec : NULL, AV_LOG_WARNING,

                   "RTP: dropping old packet received too late\n");

            return -1;

        } else if (diff <= 1) {

            /* Correct packet */

            rv = rtp_parse_packet_internal(s, pkt, buf, len);

            return rv;

        } else {

            /* Still missing some packet, enqueue this one. */

            enqueue_packet(s, buf, len);

            *bufptr = NULL;

            /* Return the first enqueued packet if the queue is full,

             * even if we're missing something */

            if (s->queue_len >= s->queue_size) {

                av_log(s->st ? s->st->codec : NULL, AV_LOG_WARNING,

                       "jitter buffer full\n");

                return rtp_parse_queued_packet(s, pkt);

            }

            return -1;

        }

    }

}
