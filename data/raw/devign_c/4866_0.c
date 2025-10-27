static int http_parse_request(HTTPContext *c)

{

    const char *p;

    char *p1;

    enum RedirType redir_type;

    char cmd[32];

    char info[1024], filename[1024];

    char url[1024], *q;

    char protocol[32];

    char msg[1024];

    const char *mime_type;

    FFServerStream *stream;

    int i;

    char ratebuf[32];

    const char *useragent = 0;



    p = c->buffer;

    get_word(cmd, sizeof(cmd), &p);

    av_strlcpy(c->method, cmd, sizeof(c->method));



    if (!strcmp(cmd, "GET"))

        c->post = 0;

    else if (!strcmp(cmd, "POST"))

        c->post = 1;

    else

        return -1;



    get_word(url, sizeof(url), &p);

    av_strlcpy(c->url, url, sizeof(c->url));



    get_word(protocol, sizeof(protocol), (const char **)&p);

    if (strcmp(protocol, "HTTP/1.0") && strcmp(protocol, "HTTP/1.1"))

        return -1;



    av_strlcpy(c->protocol, protocol, sizeof(c->protocol));



    if (config.debug)

        http_log("%s - - New connection: %s %s\n", inet_ntoa(c->from_addr.sin_addr), cmd, url);



    /* find the filename and the optional info string in the request */

    p1 = strchr(url, '?');

    if (p1) {

        av_strlcpy(info, p1, sizeof(info));

        *p1 = '\0';

    } else

        info[0] = '\0';



    av_strlcpy(filename, url + ((*url == '/') ? 1 : 0), sizeof(filename)-1);



    for (p = c->buffer; *p && *p != '\r' && *p != '\n'; ) {

        if (av_strncasecmp(p, "User-Agent:", 11) == 0) {

            useragent = p + 11;

            if (*useragent && *useragent != '\n' && av_isspace(*useragent))

                useragent++;

            break;

        }

        p = strchr(p, '\n');

        if (!p)

            break;



        p++;

    }



    redir_type = REDIR_NONE;

    if (av_match_ext(filename, "asx")) {

        redir_type = REDIR_ASX;

        filename[strlen(filename)-1] = 'f';

    } else if (av_match_ext(filename, "asf") &&

        (!useragent || av_strncasecmp(useragent, "NSPlayer", 8) != 0)) {

        /* if this isn't WMP or lookalike, return the redirector file */

        redir_type = REDIR_ASF;

    } else if (av_match_ext(filename, "rpm,ram")) {

        redir_type = REDIR_RAM;

        strcpy(filename + strlen(filename)-2, "m");

    } else if (av_match_ext(filename, "rtsp")) {

        redir_type = REDIR_RTSP;

        compute_real_filename(filename, sizeof(filename) - 1);

    } else if (av_match_ext(filename, "sdp")) {

        redir_type = REDIR_SDP;

        compute_real_filename(filename, sizeof(filename) - 1);

    }



    // "redirect" / request to index.html

    if (!strlen(filename))

        av_strlcpy(filename, "index.html", sizeof(filename) - 1);



    stream = config.first_stream;

    while (stream) {

        if (!strcmp(stream->filename, filename) && validate_acl(stream, c))

            break;

        stream = stream->next;

    }

    if (!stream) {

        snprintf(msg, sizeof(msg), "File '%s' not found", url);

        http_log("File '%s' not found\n", url);

        goto send_error;

    }



    c->stream = stream;

    memcpy(c->feed_streams, stream->feed_streams, sizeof(c->feed_streams));

    memset(c->switch_feed_streams, -1, sizeof(c->switch_feed_streams));



    if (stream->stream_type == STREAM_TYPE_REDIRECT) {

        c->http_error = 301;

        q = c->buffer;

        snprintf(q, c->buffer_size,

                      "HTTP/1.0 301 Moved\r\n"

                      "Location: %s\r\n"

                      "Content-type: text/html\r\n"

                      "\r\n"

                      "<html><head><title>Moved</title></head><body>\r\n"

                      "You should be <a href=\"%s\">redirected</a>.\r\n"

                      "</body></html>\r\n", stream->feed_filename, stream->feed_filename);

        q += strlen(q);

        /* prepare output buffer */

        c->buffer_ptr = c->buffer;

        c->buffer_end = q;

        c->state = HTTPSTATE_SEND_HEADER;

        return 0;

    }



    /* If this is WMP, get the rate information */

    if (extract_rates(ratebuf, sizeof(ratebuf), c->buffer)) {

        if (modify_current_stream(c, ratebuf)) {

            for (i = 0; i < FF_ARRAY_ELEMS(c->feed_streams); i++) {

                if (c->switch_feed_streams[i] >= 0)

                    c->switch_feed_streams[i] = -1;

            }

        }

    }



    if (c->post == 0 && stream->stream_type == STREAM_TYPE_LIVE)

        current_bandwidth += stream->bandwidth;



    /* If already streaming this feed, do not let start another feeder. */

    if (stream->feed_opened) {

        snprintf(msg, sizeof(msg), "This feed is already being received.");

        http_log("Feed '%s' already being received\n", stream->feed_filename);

        goto send_error;

    }



    if (c->post == 0 && config.max_bandwidth < current_bandwidth) {

        c->http_error = 503;

        q = c->buffer;

        snprintf(q, c->buffer_size,

                      "HTTP/1.0 503 Server too busy\r\n"

                      "Content-type: text/html\r\n"

                      "\r\n"

                      "<html><head><title>Too busy</title></head><body>\r\n"

                      "<p>The server is too busy to serve your request at this time.</p>\r\n"

                      "<p>The bandwidth being served (including your stream) is %"PRIu64"kbit/sec, "

                      "and this exceeds the limit of %"PRIu64"kbit/sec.</p>\r\n"

                      "</body></html>\r\n", current_bandwidth, config.max_bandwidth);

        q += strlen(q);

        /* prepare output buffer */

        c->buffer_ptr = c->buffer;

        c->buffer_end = q;

        c->state = HTTPSTATE_SEND_HEADER;

        return 0;

    }



    if (redir_type != REDIR_NONE) {

        const char *hostinfo = 0;



        for (p = c->buffer; *p && *p != '\r' && *p != '\n'; ) {

            if (av_strncasecmp(p, "Host:", 5) == 0) {

                hostinfo = p + 5;

                break;

            }

            p = strchr(p, '\n');

            if (!p)

                break;



            p++;

        }



        if (hostinfo) {

            char *eoh;

            char hostbuf[260];



            while (av_isspace(*hostinfo))

                hostinfo++;



            eoh = strchr(hostinfo, '\n');

            if (eoh) {

                if (eoh[-1] == '\r')

                    eoh--;



                if (eoh - hostinfo < sizeof(hostbuf) - 1) {

                    memcpy(hostbuf, hostinfo, eoh - hostinfo);

                    hostbuf[eoh - hostinfo] = 0;



                    c->http_error = 200;

                    q = c->buffer;

                    switch(redir_type) {

                    case REDIR_ASX:

                        snprintf(q, c->buffer_size,

                                      "HTTP/1.0 200 ASX Follows\r\n"

                                      "Content-type: video/x-ms-asf\r\n"

                                      "\r\n"

                                      "<ASX Version=\"3\">\r\n"

                                      //"<!-- Autogenerated by ffserver -->\r\n"

                                      "<ENTRY><REF HREF=\"http://%s/%s%s\"/></ENTRY>\r\n"

                                      "</ASX>\r\n", hostbuf, filename, info);

                        q += strlen(q);

                        break;

                    case REDIR_RAM:

                        snprintf(q, c->buffer_size,

                                      "HTTP/1.0 200 RAM Follows\r\n"

                                      "Content-type: audio/x-pn-realaudio\r\n"

                                      "\r\n"

                                      "# Autogenerated by ffserver\r\n"

                                      "http://%s/%s%s\r\n", hostbuf, filename, info);

                        q += strlen(q);

                        break;

                    case REDIR_ASF:

                        snprintf(q, c->buffer_size,

                                      "HTTP/1.0 200 ASF Redirect follows\r\n"

                                      "Content-type: video/x-ms-asf\r\n"

                                      "\r\n"

                                      "[Reference]\r\n"

                                      "Ref1=http://%s/%s%s\r\n", hostbuf, filename, info);

                        q += strlen(q);

                        break;

                    case REDIR_RTSP:

                        {

                            char hostname[256], *p;

                            /* extract only hostname */

                            av_strlcpy(hostname, hostbuf, sizeof(hostname));

                            p = strrchr(hostname, ':');

                            if (p)

                                *p = '\0';

                            snprintf(q, c->buffer_size,

                                          "HTTP/1.0 200 RTSP Redirect follows\r\n"

                                          /* XXX: incorrect MIME type ? */

                                          "Content-type: application/x-rtsp\r\n"

                                          "\r\n"

                                          "rtsp://%s:%d/%s\r\n", hostname, ntohs(config.rtsp_addr.sin_port), filename);

                            q += strlen(q);

                        }

                        break;

                    case REDIR_SDP:

                        {

                            uint8_t *sdp_data;

                            int sdp_data_size;

                            socklen_t len;

                            struct sockaddr_in my_addr;



                            snprintf(q, c->buffer_size,

                                          "HTTP/1.0 200 OK\r\n"

                                          "Content-type: application/sdp\r\n"

                                          "\r\n");

                            q += strlen(q);



                            len = sizeof(my_addr);



                            /* XXX: Should probably fail? */

                            if (getsockname(c->fd, (struct sockaddr *)&my_addr, &len))

                                http_log("getsockname() failed\n");



                            /* XXX: should use a dynamic buffer */

                            sdp_data_size = prepare_sdp_description(stream,

                                                                    &sdp_data,

                                                                    my_addr.sin_addr);

                            if (sdp_data_size > 0) {

                                memcpy(q, sdp_data, sdp_data_size);

                                q += sdp_data_size;

                                *q = '\0';

                                av_free(sdp_data);

                            }

                        }

                        break;

                    default:

                        abort();

                        break;

                    }



                    /* prepare output buffer */

                    c->buffer_ptr = c->buffer;

                    c->buffer_end = q;

                    c->state = HTTPSTATE_SEND_HEADER;

                    return 0;

                }

            }

        }



        snprintf(msg, sizeof(msg), "ASX/RAM file not handled");

        goto send_error;

    }



    stream->conns_served++;



    /* XXX: add there authenticate and IP match */



    if (c->post) {

        /* if post, it means a feed is being sent */

        if (!stream->is_feed) {

            /* However it might be a status report from WMP! Let us log the

             * data as it might come handy one day. */

            const char *logline = 0;

            int client_id = 0;



            for (p = c->buffer; *p && *p != '\r' && *p != '\n'; ) {

                if (av_strncasecmp(p, "Pragma: log-line=", 17) == 0) {

                    logline = p;

                    break;

                }

                if (av_strncasecmp(p, "Pragma: client-id=", 18) == 0)

                    client_id = strtol(p + 18, 0, 10);

                p = strchr(p, '\n');

                if (!p)

                    break;



                p++;

            }



            if (logline) {

                char *eol = strchr(logline, '\n');



                logline += 17;



                if (eol) {

                    if (eol[-1] == '\r')

                        eol--;

                    http_log("%.*s\n", (int) (eol - logline), logline);

                    c->suppress_log = 1;

                }

            }



#ifdef DEBUG

            http_log("\nGot request:\n%s\n", c->buffer);

#endif



            if (client_id && extract_rates(ratebuf, sizeof(ratebuf), c->buffer)) {

                HTTPContext *wmpc;



                /* Now we have to find the client_id */

                for (wmpc = first_http_ctx; wmpc; wmpc = wmpc->next) {

                    if (wmpc->wmp_client_id == client_id)

                        break;

                }



                if (wmpc && modify_current_stream(wmpc, ratebuf))

                    wmpc->switch_pending = 1;

            }



            snprintf(msg, sizeof(msg), "POST command not handled");

            c->stream = 0;

            goto send_error;

        }

        if (http_start_receive_data(c) < 0) {

            snprintf(msg, sizeof(msg), "could not open feed");

            goto send_error;

        }

        c->http_error = 0;

        c->state = HTTPSTATE_RECEIVE_DATA;

        return 0;

    }



#ifdef DEBUG

    if (strcmp(stream->filename + strlen(stream->filename) - 4, ".asf") == 0)

        http_log("\nGot request:\n%s\n", c->buffer);

#endif



    if (c->stream->stream_type == STREAM_TYPE_STATUS)

        goto send_status;



    /* open input stream */

    if (open_input_stream(c, info) < 0) {

        snprintf(msg, sizeof(msg), "Input stream corresponding to '%s' not found", url);

        goto send_error;

    }



    /* prepare HTTP header */

    c->buffer[0] = 0;

    av_strlcatf(c->buffer, c->buffer_size, "HTTP/1.0 200 OK\r\n");

    mime_type = c->stream->fmt->mime_type;

    if (!mime_type)

        mime_type = "application/x-octet-stream";

    av_strlcatf(c->buffer, c->buffer_size, "Pragma: no-cache\r\n");



    /* for asf, we need extra headers */

    if (!strcmp(c->stream->fmt->name,"asf_stream")) {

        /* Need to allocate a client id */



        c->wmp_client_id = av_lfg_get(&random_state);



        av_strlcatf(c->buffer, c->buffer_size, "Server: Cougar 4.1.0.3923\r\nCache-Control: no-cache\r\nPragma: client-id=%d\r\nPragma: features=\"broadcast\"\r\n", c->wmp_client_id);

    }

    av_strlcatf(c->buffer, c->buffer_size, "Content-Type: %s\r\n", mime_type);

    av_strlcatf(c->buffer, c->buffer_size, "\r\n");

    q = c->buffer + strlen(c->buffer);



    /* prepare output buffer */

    c->http_error = 0;

    c->buffer_ptr = c->buffer;

    c->buffer_end = q;

    c->state = HTTPSTATE_SEND_HEADER;

    return 0;

 send_error:

    c->http_error = 404;

    q = c->buffer;

    htmlstrip(msg);

    snprintf(q, c->buffer_size,

                  "HTTP/1.0 404 Not Found\r\n"

                  "Content-type: text/html\r\n"

                  "\r\n"

                  "<html>\n"

                  "<head><title>404 Not Found</title></head>\n"

                  "<body>%s</body>\n"

                  "</html>\n", msg);

    q += strlen(q);

    /* prepare output buffer */

    c->buffer_ptr = c->buffer;

    c->buffer_end = q;

    c->state = HTTPSTATE_SEND_HEADER;

    return 0;

 send_status:

    compute_status(c);

    c->http_error = 200; /* horrible : we use this value to avoid

                            going to the send data state */

    c->state = HTTPSTATE_SEND_HEADER;

    return 0;

}
