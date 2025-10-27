int ff_rtp_send_rtcp_feedback(RTPDemuxContext *s, URLContext *fd,

                              AVIOContext *avio)

{

    int len, need_keyframe, missing_packets;

    AVIOContext *pb;

    uint8_t *buf;

    int64_t now;

    uint16_t first_missing, missing_mask;



    if (!fd && !avio)

        return -1;



    need_keyframe = s->handler && s->handler->need_keyframe &&

                    s->handler->need_keyframe(s->dynamic_protocol_context);

    missing_packets = find_missing_packets(s, &first_missing, &missing_mask);



    if (!need_keyframe && !missing_packets)

        return 0;



    /* Send new feedback if enough time has elapsed since the last

     * feedback packet. */



    now = av_gettime();

    if (s->last_feedback_time &&

        (now - s->last_feedback_time) < MIN_FEEDBACK_INTERVAL)

        return 0;

    s->last_feedback_time = now;



    if (!fd)

        pb = avio;

    else if (avio_open_dyn_buf(&pb) < 0)

        return -1;



    if (need_keyframe) {

        avio_w8(pb, (RTP_VERSION << 6) | 1); /* PLI */

        avio_w8(pb, RTCP_PSFB);

        avio_wb16(pb, 2); /* length in words - 1 */

        // our own SSRC: we use the server's SSRC + 1 to avoid conflicts

        avio_wb32(pb, s->ssrc + 1);

        avio_wb32(pb, s->ssrc); // server SSRC

    }



    if (missing_packets) {

        avio_w8(pb, (RTP_VERSION << 6) | 1); /* NACK */

        avio_w8(pb, RTCP_RTPFB);

        avio_wb16(pb, 3); /* length in words - 1 */

        avio_wb32(pb, s->ssrc + 1);

        avio_wb32(pb, s->ssrc); // server SSRC



        avio_wb16(pb, first_missing);

        avio_wb16(pb, missing_mask);

    }



    avio_flush(pb);

    if (!fd)

        return 0;

    len = avio_close_dyn_buf(pb, &buf);

    if (len > 0 && buf) {

        ffurl_write(fd, buf, len);

        av_free(buf);

    }

    return 0;

}
