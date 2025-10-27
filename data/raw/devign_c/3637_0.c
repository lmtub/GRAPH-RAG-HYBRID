static int rtsp_fetch_packet(AVFormatContext *s, AVPacket *pkt)

{

    RTSPState *rt = s->priv_data;

    int ret, len;

    uint8_t buf[10 * RTP_MAX_PACKET_LENGTH];

    RTSPStream *rtsp_st;



    /* get next frames from the same RTP packet */

    if (rt->cur_transport_priv) {

        if (rt->transport == RTSP_TRANSPORT_RDT)

            ret = ff_rdt_parse_packet(rt->cur_transport_priv, pkt, NULL, 0);

        else

            ret = rtp_parse_packet(rt->cur_transport_priv, pkt, NULL, 0);

        if (ret == 0) {

            rt->cur_transport_priv = NULL;

            return 0;

        } else if (ret == 1) {

            return 0;

        } else {

            rt->cur_transport_priv = NULL;

        }

    }



    /* read next RTP packet */

 redo:

    switch(rt->lower_transport) {

    default:

#if CONFIG_RTSP_DEMUXER

    case RTSP_LOWER_TRANSPORT_TCP:

        len = tcp_read_packet(s, &rtsp_st, buf, sizeof(buf));

        break;

#endif

    case RTSP_LOWER_TRANSPORT_UDP:

    case RTSP_LOWER_TRANSPORT_UDP_MULTICAST:

        len = udp_read_packet(s, &rtsp_st, buf, sizeof(buf));

        if (len >=0 && rtsp_st->transport_priv && rt->transport == RTSP_TRANSPORT_RTP)

            rtp_check_and_send_back_rr(rtsp_st->transport_priv, len);

        break;

    }

    if (len < 0)

        return len;

    if (len == 0)

        return AVERROR_EOF;

    if (rt->transport == RTSP_TRANSPORT_RDT)

        ret = ff_rdt_parse_packet(rtsp_st->transport_priv, pkt, buf, len);

    else

        ret = rtp_parse_packet(rtsp_st->transport_priv, pkt, buf, len);

    if (ret < 0)

        goto redo;

    if (ret == 1) {

        /* more packets may follow, so we save the RTP context */

        rt->cur_transport_priv = rtsp_st->transport_priv;

    }



    return ret;

}
