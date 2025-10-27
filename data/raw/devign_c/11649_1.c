make_setup_request (AVFormatContext *s, const char *host, int port,

                    int lower_transport, const char *real_challenge)

{

    RTSPState *rt = s->priv_data;

    int j, i, err;

    RTSPStream *rtsp_st;

    RTSPHeader reply1, *reply = &reply1;

    char cmd[2048];

    const char *trans_pref;



    if (rt->server_type == RTSP_SERVER_REAL)

        trans_pref = "x-pn-tng";

    else

        trans_pref = "RTP/AVP";



    /* for each stream, make the setup request */

    /* XXX: we assume the same server is used for the control of each

       RTSP stream */



    for(j = RTSP_RTP_PORT_MIN, i = 0; i < rt->nb_rtsp_streams; ++i) {

        char transport[2048];



        rtsp_st = rt->rtsp_streams[i];



        /* RTP/UDP */

        if (lower_transport == RTSP_LOWER_TRANSPORT_UDP) {

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

            snprintf(transport, sizeof(transport) - 1,

                     "%s/UDP;unicast;client_port=%d",

                     trans_pref, port);

            if (rt->server_type == RTSP_SERVER_RTP)

                av_strlcatf(transport, sizeof(transport), "-%d", port + 1);

        }



        /* RTP/TCP */

        else if (lower_transport == RTSP_LOWER_TRANSPORT_TCP) {

            snprintf(transport, sizeof(transport) - 1,

                     "%s/TCP", trans_pref);

        }



        else if (lower_transport == RTSP_LOWER_TRANSPORT_UDP_MULTICAST) {

            snprintf(transport, sizeof(transport) - 1,

                     "%s/UDP;multicast", trans_pref);

        }

        if (rt->server_type == RTSP_SERVER_REAL)

            av_strlcat(transport, ";mode=play", sizeof(transport));

        snprintf(cmd, sizeof(cmd),

                 "SETUP %s RTSP/1.0\r\n"

                 "Transport: %s\r\n",

                 rtsp_st->control_url, transport);

        if (i == 0 && rt->server_type == RTSP_SERVER_REAL) {

            char real_res[41], real_csum[9];

            ff_rdt_calc_response_and_checksum(real_res, real_csum,

                                              real_challenge);

            av_strlcatf(cmd, sizeof(cmd),

                        "If-Match: %s\r\n"

                        "RealChallenge2: %s, sd=%s\r\n",

                        rt->session_id, real_res, real_csum);

        }

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

            if (reply->transports[0].lower_transport != rt->lower_transport) {

                err = AVERROR_INVALIDDATA;

                goto fail;

            }

        } else {

            rt->lower_transport = reply->transports[0].lower_transport;

        }



        /* close RTP connection if not choosen */

        if (reply->transports[0].lower_transport != RTSP_LOWER_TRANSPORT_UDP &&

            (lower_transport == RTSP_LOWER_TRANSPORT_UDP)) {

            url_close(rtsp_st->rtp_handle);

            rtsp_st->rtp_handle = NULL;

        }



        switch(reply->transports[0].lower_transport) {

        case RTSP_LOWER_TRANSPORT_TCP:

            rtsp_st->interleaved_min = reply->transports[0].interleaved_min;

            rtsp_st->interleaved_max = reply->transports[0].interleaved_max;

            break;



        case RTSP_LOWER_TRANSPORT_UDP:

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

        case RTSP_LOWER_TRANSPORT_UDP_MULTICAST:

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



        if ((err = rtsp_open_transport_ctx(s, rtsp_st)))

            goto fail;

    }



    if (rt->server_type == RTSP_SERVER_REAL)

        rt->need_subscription = 1;



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
