static void rtsp_parse_transport(RTSPMessageHeader *reply, const char *p)

{

    char transport_protocol[16];

    char profile[16];

    char lower_transport[16];

    char parameter[16];

    RTSPTransportField *th;

    char buf[256];



    reply->nb_transports = 0;



    for(;;) {

        skip_spaces(&p);

        if (*p == '\0')

            break;



        th = &reply->transports[reply->nb_transports];



        get_word_sep(transport_protocol, sizeof(transport_protocol),

                     "/", &p);

        if (!strcasecmp (transport_protocol, "rtp")) {

            get_word_sep(profile, sizeof(profile), "/;,", &p);

            lower_transport[0] = '\0';

            /* rtp/avp/<protocol> */

            if (*p == '/') {

                get_word_sep(lower_transport, sizeof(lower_transport),

                             ";,", &p);

            }

            th->transport = RTSP_TRANSPORT_RTP;

        } else if (!strcasecmp (transport_protocol, "x-pn-tng") ||

                   !strcasecmp (transport_protocol, "x-real-rdt")) {

            /* x-pn-tng/<protocol> */

            get_word_sep(lower_transport, sizeof(lower_transport), "/;,", &p);

            profile[0] = '\0';

            th->transport = RTSP_TRANSPORT_RDT;

        }

        if (!strcasecmp(lower_transport, "TCP"))

            th->lower_transport = RTSP_LOWER_TRANSPORT_TCP;

        else

            th->lower_transport = RTSP_LOWER_TRANSPORT_UDP;



        if (*p == ';')

            p++;

        /* get each parameter */

        while (*p != '\0' && *p != ',') {

            get_word_sep(parameter, sizeof(parameter), "=;,", &p);

            if (!strcmp(parameter, "port")) {

                if (*p == '=') {

                    p++;

                    rtsp_parse_range(&th->port_min, &th->port_max, &p);

                }

            } else if (!strcmp(parameter, "client_port")) {

                if (*p == '=') {

                    p++;

                    rtsp_parse_range(&th->client_port_min,

                                     &th->client_port_max, &p);

                }

            } else if (!strcmp(parameter, "server_port")) {

                if (*p == '=') {

                    p++;

                    rtsp_parse_range(&th->server_port_min,

                                     &th->server_port_max, &p);

                }

            } else if (!strcmp(parameter, "interleaved")) {

                if (*p == '=') {

                    p++;

                    rtsp_parse_range(&th->interleaved_min,

                                     &th->interleaved_max, &p);

                }

            } else if (!strcmp(parameter, "multicast")) {

                if (th->lower_transport == RTSP_LOWER_TRANSPORT_UDP)

                    th->lower_transport = RTSP_LOWER_TRANSPORT_UDP_MULTICAST;

            } else if (!strcmp(parameter, "ttl")) {

                if (*p == '=') {

                    p++;

                    th->ttl = strtol(p, (char **)&p, 10);

                }

            } else if (!strcmp(parameter, "destination")) {

                struct in_addr ipaddr;



                if (*p == '=') {

                    p++;

                    get_word_sep(buf, sizeof(buf), ";,", &p);

                    if (inet_aton(buf, &ipaddr))

                        th->destination = ntohl(ipaddr.s_addr);

                }

            }

            while (*p != ';' && *p != '\0' && *p != ',')

                p++;

            if (*p == ';')

                p++;

        }

        if (*p == ',')

            p++;



        reply->nb_transports++;

    }

}
