int ff_rtsp_connect(AVFormatContext *s)

{

    RTSPState *rt = s->priv_data;

    char host[1024], path[1024], tcpname[1024], cmd[2048], auth[128];

    char *option_list, *option, *filename;

    URLContext *rtsp_hd, *rtsp_hd_out;

    int port, err, tcp_fd;

    RTSPMessageHeader reply1 = {}, *reply = &reply1;

    int lower_transport_mask = 0;

    char real_challenge[64];

    struct sockaddr_storage peer;

    socklen_t peer_len = sizeof(peer);



    if (!ff_network_init())

        return AVERROR(EIO);

redirect:

    rt->control_transport = RTSP_MODE_PLAIN;

    /* extract hostname and port */

    ff_url_split(NULL, 0, auth, sizeof(auth),

                 host, sizeof(host), &port, path, sizeof(path), s->filename);

    if (*auth) {

        av_strlcpy(rt->auth, auth, sizeof(rt->auth));

    }

    if (port < 0)

        port = RTSP_DEFAULT_PORT;



    /* search for options */

    option_list = strrchr(path, '?');

    if (option_list) {

        /* Strip out the RTSP specific options, write out the rest of

         * the options back into the same string. */

        filename = option_list;

        while (option_list) {

            /* move the option pointer */

            option = ++option_list;

            option_list = strchr(option_list, '&');

            if (option_list)

                *option_list = 0;



            /* handle the options */

            if (!strcmp(option, "udp")) {

                lower_transport_mask |= (1<< RTSP_LOWER_TRANSPORT_UDP);

            } else if (!strcmp(option, "multicast")) {

                lower_transport_mask |= (1<< RTSP_LOWER_TRANSPORT_UDP_MULTICAST);

            } else if (!strcmp(option, "tcp")) {

                lower_transport_mask |= (1<< RTSP_LOWER_TRANSPORT_TCP);

            } else if(!strcmp(option, "http")) {

                lower_transport_mask |= (1<< RTSP_LOWER_TRANSPORT_TCP);

                rt->control_transport = RTSP_MODE_TUNNEL;

            } else {

                /* Write options back into the buffer, using memmove instead

                 * of strcpy since the strings may overlap. */

                int len = strlen(option);

                memmove(++filename, option, len);

                filename += len;

                if (option_list) *filename = '&';

            }

        }

        *filename = 0;

    }



    if (!lower_transport_mask)

        lower_transport_mask = (1 << RTSP_LOWER_TRANSPORT_NB) - 1;



    if (s->oformat) {

        /* Only UDP or TCP - UDP multicast isn't supported. */

        lower_transport_mask &= (1 << RTSP_LOWER_TRANSPORT_UDP) |

                                (1 << RTSP_LOWER_TRANSPORT_TCP);

        if (!lower_transport_mask || rt->control_transport == RTSP_MODE_TUNNEL) {

            av_log(s, AV_LOG_ERROR, "Unsupported lower transport method, "

                                    "only UDP and TCP are supported for output.\n");

            err = AVERROR(EINVAL);

            goto fail;

        }

    }



    /* Construct the URI used in request; this is similar to s->filename,

     * but with authentication credentials removed and RTSP specific options

     * stripped out. */

    ff_url_join(rt->control_uri, sizeof(rt->control_uri), "rtsp", NULL,

                host, port, "%s", path);



    if (rt->control_transport == RTSP_MODE_TUNNEL) {

        /* set up initial handshake for tunneling */

        char httpname[1024];

        char sessioncookie[17];

        char headers[1024];



        ff_url_join(httpname, sizeof(httpname), "http", NULL, host, port, "%s", path);

        snprintf(sessioncookie, sizeof(sessioncookie), "%08x%08x",

                 av_get_random_seed(), av_get_random_seed());



        /* GET requests */

        if (url_open(&rtsp_hd, httpname, URL_RDONLY) < 0) {

            err = AVERROR(EIO);

            goto fail;

        }



        /* generate GET headers */

        snprintf(headers, sizeof(headers),

                 "x-sessioncookie: %s\r\n"

                 "Accept: application/x-rtsp-tunnelled\r\n"

                 "Pragma: no-cache\r\n"

                 "Cache-Control: no-cache\r\n",

                 sessioncookie);

        ff_http_set_headers(rtsp_hd, headers);



        /* complete the connection */

        if (url_read(rtsp_hd, NULL, 0)) {


            err = AVERROR(EIO);

            goto fail;

        }



        /* POST requests */

        if (url_open(&rtsp_hd_out, httpname, URL_WRONLY) < 0 ) {


            err = AVERROR(EIO);

            goto fail;

        }



        /* generate POST headers */

        snprintf(headers, sizeof(headers),

                 "x-sessioncookie: %s\r\n"

                 "Content-Type: application/x-rtsp-tunnelled\r\n"

                 "Pragma: no-cache\r\n"

                 "Cache-Control: no-cache\r\n"

                 "Content-Length: 32767\r\n"

                 "Expires: Sun, 9 Jan 1972 00:00:00 GMT\r\n",

                 sessioncookie);

        ff_http_set_headers(rtsp_hd_out, headers);

        ff_http_set_chunked_transfer_encoding(rtsp_hd_out, 0);



    } else {

        /* open the tcp connection */

        ff_url_join(tcpname, sizeof(tcpname), "tcp", NULL, host, port, NULL);

        if (url_open(&rtsp_hd, tcpname, URL_RDWR) < 0) {

            err = AVERROR(EIO);

            goto fail;

        }

        rtsp_hd_out = rtsp_hd;

    }

    rt->rtsp_hd = rtsp_hd;

    rt->rtsp_hd_out = rtsp_hd_out;

    rt->seq = 0;



    tcp_fd = url_get_file_handle(rtsp_hd);

    if (!getpeername(tcp_fd, (struct sockaddr*) &peer, &peer_len)) {

        getnameinfo((struct sockaddr*) &peer, peer_len, host, sizeof(host),

                    NULL, 0, NI_NUMERICHOST);

    }



    /* request options supported by the server; this also detects server

     * type */

    for (rt->server_type = RTSP_SERVER_RTP;;) {

        cmd[0] = 0;

        if (rt->server_type == RTSP_SERVER_REAL)

            av_strlcat(cmd,

                       /**

                        * The following entries are required for proper

                        * streaming from a Realmedia server. They are

                        * interdependent in some way although we currently

                        * don't quite understand how. Values were copied

                        * from mplayer SVN r23589.

                        * @param CompanyID is a 16-byte ID in base64

                        * @param ClientChallenge is a 16-byte ID in hex

                        */

                       "ClientChallenge: 9e26d33f2984236010ef6253fb1887f7\r\n"

                       "PlayerStarttime: [28/03/2003:22:50:23 00:00]\r\n"

                       "CompanyID: KnKV4M4I/B2FjJ1TToLycw==\r\n"

                       "GUID: 00000000-0000-0000-0000-000000000000\r\n",

                       sizeof(cmd));

        ff_rtsp_send_cmd(s, "OPTIONS", rt->control_uri, cmd, reply, NULL);

        if (reply->status_code != RTSP_STATUS_OK) {

            err = AVERROR_INVALIDDATA;

            goto fail;

        }



        /* detect server type if not standard-compliant RTP */

        if (rt->server_type != RTSP_SERVER_REAL && reply->real_challenge[0]) {

            rt->server_type = RTSP_SERVER_REAL;

            continue;

        } else if (!strncasecmp(reply->server, "WMServer/", 9)) {

            rt->server_type = RTSP_SERVER_WMS;

        } else if (rt->server_type == RTSP_SERVER_REAL)

            strcpy(real_challenge, reply->real_challenge);

        break;

    }



    if (s->iformat)

        err = rtsp_setup_input_streams(s, reply);

    else

        err = rtsp_setup_output_streams(s, host);

    if (err)

        goto fail;



    do {

        int lower_transport = ff_log2_tab[lower_transport_mask &

                                  ~(lower_transport_mask - 1)];



        err = make_setup_request(s, host, port, lower_transport,

                                 rt->server_type == RTSP_SERVER_REAL ?

                                     real_challenge : NULL);

        if (err < 0)

            goto fail;

        lower_transport_mask &= ~(1 << lower_transport);

        if (lower_transport_mask == 0 && err == 1) {

            err = FF_NETERROR(EPROTONOSUPPORT);

            goto fail;

        }

    } while (err);



    rt->state = RTSP_STATE_IDLE;

    rt->seek_timestamp = 0; /* default is to start stream at position zero */

    return 0;

 fail:

    ff_rtsp_close_streams(s);

    ff_rtsp_close_connections(s);

    if (reply->status_code >=300 && reply->status_code < 400 && s->iformat) {

        av_strlcpy(s->filename, reply->location, sizeof(s->filename));

        av_log(s, AV_LOG_INFO, "Status %d: Redirecting to %s\n",

               reply->status_code,

               s->filename);

        goto redirect;

    }

    ff_network_close();

    return err;

}