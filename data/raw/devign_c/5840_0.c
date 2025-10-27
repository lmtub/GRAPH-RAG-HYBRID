make_setup_request (AVFormatContext *s, const char *host, int port, int protocol)

{

    RTSPState *rt = s->priv_data;

    int j, i, err;

    RTSPStream *rtsp_st;

    AVStream *st;

    RTSPHeader reply1, *reply = &reply1;

    char cmd[2048];



    /* for each stream, make the setup request */

    /* XXX: we assume the same server is used for the control of each

       RTSP stream */



    for(j = RTSP_RTP_PORT_MIN, i = 0; i < rt->nb_rtsp_streams; ++i) {

        char transport[2048];



        rtsp_st = rt->rtsp_streams[i];



        /* compute available transports */

        transport[0] = '\0';



        /* RTP/UDP */

        if (protocol == RTSP_PROTOCOL_RTP_UDP) {

            char buf[256];



            /* first try in specified port range */

            if (RTSP_RTP_PORT_MIN != 0) {

                while(j <= RTSP_RTP_PORT_MAX) {

                    snprintf(buf, sizeof(buf), "rtp://%s?localport=%d", host, j);

                    j += 2; /* we will use two port by rtp stream (rtp and rtcp) */

                    if (url_open(&rtsp_st->rtp_handle, buf, URL_RDWR) == 0) {

                        goto rtp_opened;

                    }

                }

            }



/*            then try on any port

**            if (url_open(&rtsp_st->rtp_handle, "rtp://", URL_RDONLY) < 0) {

**                err = AVERROR_INVALIDDATA;

**                goto fail;

**            }

*/



        rtp_opened:

            port = rtp_get_local_port(rtsp_st->rtp_handle);

            if (transport[0] != '\0')

                av_strlcat(transport, ",", sizeof(transport));

            snprintf(transport + strlen(transport), sizeof(transport) - strlen(transport) - 1,

                     "RTP/AVP/UDP;unicast;client_port=%d-%d",

                     port, port + 1);

        }



        /* RTP/TCP */

        else if (protocol == RTSP_PROTOCOL_RTP_TCP) {

            if (transport[0] != '\0')

                av_strlcat(transport, ",", sizeof(transport));

            snprintf(transport + strlen(transport), sizeof(transport) - strlen(transport) - 1,

                     "RTP/AVP/TCP");

        }



        else if (protocol == RTSP_PROTOCOL_RTP_UDP_MULTICAST) {

            if (transport[0] != '\0')

                av_strlcat(transport, ",", sizeof(transport));

            snprintf(transport + strlen(transport),

                     sizeof(transport) - strlen(transport) - 1,

                     "RTP/AVP/UDP;multicast");

        }

        snprintf(cmd, sizeof(cmd),

                 "SETUP %s RTSP/1.0\r\n"

                 "Transport: %s\r\n",

                 rtsp_st->control_url, transport);

        rtsp_send_cmd(s, cmd, reply, NULL);

        if (reply->status_code == 461 /* Unsupported protocol */ && i == 0) {

            err = 1;

            goto fail;

        } else if (reply->status_code != RTSP_STATUS_OK ||

                   reply->nb_transports != 1) {

            err = AVERROR_INVALIDDATA;

            goto fail;

        }



        /* XXX: same protocol for all streams is required */

        if (i > 0) {

            if (reply->transports[0].protocol != rt->protocol) {

                err = AVERROR_INVALIDDATA;

                goto fail;

            }

        } else {

            rt->protocol = reply->transports[0].protocol;

        }



        /* close RTP connection if not choosen */

        if (reply->transports[0].protocol != RTSP_PROTOCOL_RTP_UDP &&

            (protocol == RTSP_PROTOCOL_RTP_UDP)) {

            url_close(rtsp_st->rtp_handle);

            rtsp_st->rtp_handle = NULL;

        }



        switch(reply->transports[0].protocol) {

        case RTSP_PROTOCOL_RTP_TCP:

            rtsp_st->interleaved_min = reply->transports[0].interleaved_min;

            rtsp_st->interleaved_max = reply->transports[0].interleaved_max;

            break;



        case RTSP_PROTOCOL_RTP_UDP:

            {

                char url[1024];



                /* XXX: also use address if specified */

                snprintf(url, sizeof(url), "rtp://%s:%d",

                         host, reply->transports[0].server_port_min);

                if (rtp_set_remote_url(rtsp_st->rtp_handle, url) < 0) {

                    err = AVERROR_INVALIDDATA;

                    goto fail;

                }

            }

            break;

        case RTSP_PROTOCOL_RTP_UDP_MULTICAST:

            {

                char url[1024];

                struct in_addr in;



                in.s_addr = htonl(reply->transports[0].destination);

                snprintf(url, sizeof(url), "rtp://%s:%d?ttl=%d",

                         inet_ntoa(in),

                         reply->transports[0].port_min,

                         reply->transports[0].ttl);

                if (url_open(&rtsp_st->rtp_handle, url, URL_RDWR) < 0) {

                    err = AVERROR_INVALIDDATA;

                    goto fail;

                }

            }

            break;

        }

        /* open the RTP context */

        st = NULL;

        if (rtsp_st->stream_index >= 0)

            st = s->streams[rtsp_st->stream_index];

        if (!st)

            s->ctx_flags |= AVFMTCTX_NOHEADER;

        rtsp_st->rtp_ctx = rtp_parse_open(s, st, rtsp_st->rtp_handle, rtsp_st->sdp_payload_type, &rtsp_st->rtp_payload_data);



        if (!rtsp_st->rtp_ctx) {

            err = AVERROR(ENOMEM);

            goto fail;

        } else {

            if(rtsp_st->dynamic_handler) {

                rtsp_st->rtp_ctx->dynamic_protocol_context= rtsp_st->dynamic_protocol_context;

                rtsp_st->rtp_ctx->parse_packet= rtsp_st->dynamic_handler->parse_packet;

            }

        }

    }



    return 0;



fail:

    for (i=0; i<rt->nb_rtsp_streams; i++) {

        if (rt->rtsp_streams[i]->rtp_handle) {

            url_close(rt->rtsp_streams[i]->rtp_handle);

            rt->rtsp_streams[i]->rtp_handle = NULL;

        }

    }

    return err;

}
