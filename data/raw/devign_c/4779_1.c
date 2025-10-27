int ff_rtsp_fetch_packet(AVFormatContext *s, AVPacket *pkt)

{

    RTSPState *rt = s->priv_data;

    int ret, len;

    RTSPStream *rtsp_st, *first_queue_st = NULL;

    int64_t wait_end = 0;



    if (rt->nb_byes == rt->nb_rtsp_streams)

        return AVERROR_EOF;



    /* get next frames from the same RTP packet */

    if (rt->cur_transport_priv) {

        if (rt->transport == RTSP_TRANSPORT_RDT) {

            ret = ff_rdt_parse_packet(rt->cur_transport_priv, pkt, NULL, 0);

        } else if (rt->transport == RTSP_TRANSPORT_RTP) {

            ret = ff_rtp_parse_packet(rt->cur_transport_priv, pkt, NULL, 0);

        } else if (rt->ts && CONFIG_RTPDEC) {

            ret = ff_mpegts_parse_packet(rt->ts, pkt, rt->recvbuf + rt->recvbuf_pos, rt->recvbuf_len - rt->recvbuf_pos);

            if (ret >= 0) {

                rt->recvbuf_pos += ret;

                ret = rt->recvbuf_pos < rt->recvbuf_len;

            }

        } else

            ret = -1;

        if (ret == 0) {

            rt->cur_transport_priv = NULL;

            return 0;

        } else if (ret == 1) {

            return 0;

        } else

            rt->cur_transport_priv = NULL;

    }



redo:

    if (rt->transport == RTSP_TRANSPORT_RTP) {

        int i;

        int64_t first_queue_time = 0;

        for (i = 0; i < rt->nb_rtsp_streams; i++) {

            RTPDemuxContext *rtpctx = rt->rtsp_streams[i]->transport_priv;

            int64_t queue_time;

            if (!rtpctx)

                continue;

            queue_time = ff_rtp_queued_packet_time(rtpctx);

            if (queue_time && (queue_time - first_queue_time < 0 ||

                               !first_queue_time)) {

                first_queue_time = queue_time;

                first_queue_st   = rt->rtsp_streams[i];

            }

        }

        if (first_queue_time) {

            wait_end = first_queue_time + s->max_delay;

        } else {

            wait_end = 0;

            first_queue_st = NULL;

        }

    }



    /* read next RTP packet */

    if (!rt->recvbuf) {

        rt->recvbuf = av_malloc(RECVBUF_SIZE);

        if (!rt->recvbuf)

            return AVERROR(ENOMEM);

    }



    switch(rt->lower_transport) {

    default:

#if CONFIG_RTSP_DEMUXER

    case RTSP_LOWER_TRANSPORT_TCP:

        len = ff_rtsp_tcp_read_packet(s, &rtsp_st, rt->recvbuf, RECVBUF_SIZE);

        break;

#endif

    case RTSP_LOWER_TRANSPORT_UDP:

    case RTSP_LOWER_TRANSPORT_UDP_MULTICAST:

        len = udp_read_packet(s, &rtsp_st, rt->recvbuf, RECVBUF_SIZE, wait_end);

        if (len > 0 && rtsp_st->transport_priv && rt->transport == RTSP_TRANSPORT_RTP)

            ff_rtp_check_and_send_back_rr(rtsp_st->transport_priv, rtsp_st->rtp_handle, NULL, len);

        break;

    case RTSP_LOWER_TRANSPORT_CUSTOM:

        if (first_queue_st && rt->transport == RTSP_TRANSPORT_RTP &&

            wait_end && wait_end < av_gettime_relative())

            len = AVERROR(EAGAIN);

        else

            len = ffio_read_partial(s->pb, rt->recvbuf, RECVBUF_SIZE);

        len = pick_stream(s, &rtsp_st, rt->recvbuf, len);

        if (len > 0 && rtsp_st->transport_priv && rt->transport == RTSP_TRANSPORT_RTP)

            ff_rtp_check_and_send_back_rr(rtsp_st->transport_priv, NULL, s->pb, len);

        break;

    }

    if (len == AVERROR(EAGAIN) && first_queue_st &&

        rt->transport == RTSP_TRANSPORT_RTP) {

        rtsp_st = first_queue_st;

        ret = ff_rtp_parse_packet(rtsp_st->transport_priv, pkt, NULL, 0);

        goto end;

    }

    if (len < 0)

        return len;

    if (len == 0)

        return AVERROR_EOF;

    if (rt->transport == RTSP_TRANSPORT_RDT) {

        ret = ff_rdt_parse_packet(rtsp_st->transport_priv, pkt, &rt->recvbuf, len);

    } else if (rt->transport == RTSP_TRANSPORT_RTP) {

        ret = ff_rtp_parse_packet(rtsp_st->transport_priv, pkt, &rt->recvbuf, len);

        if (rtsp_st->feedback) {

            AVIOContext *pb = NULL;

            if (rt->lower_transport == RTSP_LOWER_TRANSPORT_CUSTOM)

                pb = s->pb;

            ff_rtp_send_rtcp_feedback(rtsp_st->transport_priv, rtsp_st->rtp_handle, pb);

        }

        if (ret < 0) {

            /* Either bad packet, or a RTCP packet. Check if the

             * first_rtcp_ntp_time field was initialized. */

            RTPDemuxContext *rtpctx = rtsp_st->transport_priv;

            if (rtpctx->first_rtcp_ntp_time != AV_NOPTS_VALUE) {

                /* first_rtcp_ntp_time has been initialized for this stream,

                 * copy the same value to all other uninitialized streams,

                 * in order to map their timestamp origin to the same ntp time

                 * as this one. */

                int i;

                AVStream *st = NULL;

                if (rtsp_st->stream_index >= 0)

                    st = s->streams[rtsp_st->stream_index];

                for (i = 0; i < rt->nb_rtsp_streams; i++) {

                    RTPDemuxContext *rtpctx2 = rt->rtsp_streams[i]->transport_priv;

                    AVStream *st2 = NULL;

                    if (rt->rtsp_streams[i]->stream_index >= 0)

                        st2 = s->streams[rt->rtsp_streams[i]->stream_index];

                    if (rtpctx2 && st && st2 &&

                        rtpctx2->first_rtcp_ntp_time == AV_NOPTS_VALUE) {

                        rtpctx2->first_rtcp_ntp_time = rtpctx->first_rtcp_ntp_time;

                        rtpctx2->rtcp_ts_offset = av_rescale_q(

                            rtpctx->rtcp_ts_offset, st->time_base,

                            st2->time_base);

                    }

                }

            }

            if (ret == -RTCP_BYE) {

                rt->nb_byes++;



                av_log(s, AV_LOG_DEBUG, "Received BYE for stream %d (%d/%d)\n",

                       rtsp_st->stream_index, rt->nb_byes, rt->nb_rtsp_streams);



                if (rt->nb_byes == rt->nb_rtsp_streams)

                    return AVERROR_EOF;

            }

        }

    } else if (rt->ts && CONFIG_RTPDEC) {

        ret = ff_mpegts_parse_packet(rt->ts, pkt, rt->recvbuf, len);

        if (ret >= 0) {

            if (ret < len) {

                rt->recvbuf_len = len;

                rt->recvbuf_pos = ret;

                rt->cur_transport_priv = rt->ts;

                return 1;

            } else {

                ret = 0;

            }

        }

    } else {

        return AVERROR_INVALIDDATA;

    }

end:

    if (ret < 0)

        goto redo;

    if (ret == 1)

        /* more packets may follow, so we save the RTP context */

        rt->cur_transport_priv = rtsp_st->transport_priv;



    return ret;

}
