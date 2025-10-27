static int http_parse_request(HTTPContext *c)

{

    char *p;

    int post;

    char cmd[32];

    char info[1024], *filename;

    char url[1024], *q;

    char protocol[32];

    char msg[1024];

    const char *mime_type;

    FFStream *stream;



    p = c->buffer;

    q = cmd;

    while (!isspace(*p) && *p != '\0') {

        if ((q - cmd) < sizeof(cmd) - 1)

            *q++ = *p;

        p++;

    }

    *q = '\0';

    if (!strcmp(cmd, "GET"))

        post = 0;

    else if (!strcmp(cmd, "POST"))

        post = 1;

    else

        return -1;



    while (isspace(*p)) p++;

    q = url;

    while (!isspace(*p) && *p != '\0') {

        if ((q - url) < sizeof(url) - 1)

            *q++ = *p;

        p++;

    }

    *q = '\0';



    while (isspace(*p)) p++;

    q = protocol;

    while (!isspace(*p) && *p != '\0') {

        if ((q - protocol) < sizeof(protocol) - 1)

            *q++ = *p;

        p++;

    }

    *q = '\0';

    if (strcmp(protocol, "HTTP/1.0") && strcmp(protocol, "HTTP/1.1"))

        return -1;

    

    /* find the filename and the optional info string in the request */

    p = url;

    if (*p == '/')

        p++;

    filename = p;

    p = strchr(p, '?');

    if (p) {

        strcpy(info, p);

        *p = '\0';

    } else {

        info[0] = '\0';

    }



    stream = first_stream;

    while (stream != NULL) {

        if (!strcmp(stream->filename, filename))

            break;

        stream = stream->next;

    }

    if (stream == NULL) {

        sprintf(msg, "File '%s' not found", url);

        goto send_error;

    }

    c->stream = stream;

    

    /* should do it after so that the size can be computed */

    {

        char buf1[32], buf2[32], *p;

        time_t ti;

        /* XXX: reentrant function ? */

        p = inet_ntoa(c->from_addr.sin_addr);

        strcpy(buf1, p);

        ti = time(NULL);

        p = ctime(&ti);

        strcpy(buf2, p);

        p = buf2 + strlen(p) - 1;

        if (*p == '\n')

            *p = '\0';

        http_log("%s - - [%s] \"%s %s %s\" %d %d\n", 

                 buf1, buf2, cmd, url, protocol, 200, 1024);

    }



    /* XXX: add there authenticate and IP match */



    if (post) {

        /* if post, it means a feed is being sent */

        if (!stream->is_feed) {

            sprintf(msg, "POST command not handled");

            goto send_error;

        }

        if (http_start_receive_data(c) < 0) {

            sprintf(msg, "could not open feed");

            goto send_error;

        }

        c->http_error = 0;

        c->state = HTTPSTATE_RECEIVE_DATA;

        return 0;

    }



    if (c->stream->stream_type == STREAM_TYPE_STATUS)

        goto send_stats;



    /* open input stream */

    if (open_input_stream(c, info) < 0) {

        sprintf(msg, "Input stream corresponding to '%s' not found", url);

        goto send_error;

    }



    /* prepare http header */

    q = c->buffer;

    q += sprintf(q, "HTTP/1.0 200 OK\r\n");

    mime_type = c->stream->fmt->mime_type;

    if (!mime_type)

        mime_type = "application/x-octet_stream";

    q += sprintf(q, "Pragma: no-cache\r\n");



    /* for asf, we need extra headers */

    if (!strcmp(c->stream->fmt->name,"asf")) {

        q += sprintf(q, "Server: Cougar 4.1.0.3923\r\nCache-Control: no-cache\r\nPragma: client-id=1234\r\nPragma: features=\"broadcast\"\r\n");

        mime_type = "application/octet-stream";

    }

    q += sprintf(q, "Content-Type: %s\r\n", mime_type);

    q += sprintf(q, "\r\n");

    

    /* prepare output buffer */

    c->http_error = 0;

    c->buffer_ptr = c->buffer;

    c->buffer_end = q;

    c->state = HTTPSTATE_SEND_HEADER;

    return 0;

 send_error:

    c->http_error = 404;

    q = c->buffer;

    q += sprintf(q, "HTTP/1.0 404 Not Found\r\n");

    q += sprintf(q, "Content-type: %s\r\n", "text/html");

    q += sprintf(q, "\r\n");

    q += sprintf(q, "<HTML>\n");

    q += sprintf(q, "<HEAD><TITLE>404 Not Found</TITLE></HEAD>\n");

    q += sprintf(q, "<BODY>%s</BODY>\n", msg);

    q += sprintf(q, "</HTML>\n");



    /* prepare output buffer */

    c->buffer_ptr = c->buffer;

    c->buffer_end = q;

    c->state = HTTPSTATE_SEND_HEADER;

    return 0;

 send_stats:

    compute_stats(c);

    c->http_error = 200; /* horrible : we use this value to avoid

                            going to the send data state */

    c->state = HTTPSTATE_SEND_HEADER;

    return 0;

}
