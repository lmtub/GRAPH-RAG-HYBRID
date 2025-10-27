static int process_line(URLContext *h, char *line, int line_count,

                        int *new_location)

{

    HTTPContext *s = h->priv_data;

    char *tag, *p, *end;



    /* end of header */

    if (line[0] == '\0') {

        s->end_header = 1;

        return 0;

    }



    p = line;

    if (line_count == 0) {

        while (!isspace(*p) && *p != '\0')

            p++;

        while (isspace(*p))

            p++;

        s->http_code = strtol(p, &end, 10);



        av_dlog(NULL, "http_code=%d\n", s->http_code);



        /* error codes are 4xx and 5xx, but regard 401 as a success, so we

         * don't abort until all headers have been parsed. */

        if (s->http_code >= 400 && s->http_code < 600 && (s->http_code != 401

            || s->auth_state.auth_type != HTTP_AUTH_NONE) &&

            (s->http_code != 407 || s->proxy_auth_state.auth_type != HTTP_AUTH_NONE)) {

            end += strspn(end, SPACE_CHARS);

            av_log(h, AV_LOG_WARNING, "HTTP error %d %s\n",

                   s->http_code, end);

            return -1;

        }

    } else {

        while (*p != '\0' && *p != ':')

            p++;

        if (*p != ':')

            return 1;



        *p = '\0';

        tag = line;

        p++;

        while (isspace(*p))

            p++;

        if (!av_strcasecmp(tag, "Location")) {

            av_strlcpy(s->location, p, sizeof(s->location));

            *new_location = 1;

        } else if (!av_strcasecmp (tag, "Content-Length") && s->filesize == -1) {

            s->filesize = strtoll(p, NULL, 10);

        } else if (!av_strcasecmp (tag, "Content-Range")) {

            /* "bytes $from-$to/$document_size" */

            const char *slash;

            if (!strncmp (p, "bytes ", 6)) {

                p += 6;

                s->off = strtoll(p, NULL, 10);

                if ((slash = strchr(p, '/')) && strlen(slash) > 0)

                    s->filesize = strtoll(slash+1, NULL, 10);

            }

            if (s->seekable == -1 && (!s->is_akamai || s->filesize != 2147483647))

                h->is_streamed = 0; /* we _can_ in fact seek */

        } else if (!av_strcasecmp(tag, "Accept-Ranges") && !strncmp(p, "bytes", 5) && s->seekable == -1) {

            h->is_streamed = 0;

        } else if (!av_strcasecmp (tag, "Transfer-Encoding") && !av_strncasecmp(p, "chunked", 7)) {

            s->filesize = -1;

            s->chunksize = 0;

        } else if (!av_strcasecmp (tag, "WWW-Authenticate")) {

            ff_http_auth_handle_header(&s->auth_state, tag, p);

        } else if (!av_strcasecmp (tag, "Authentication-Info")) {

            ff_http_auth_handle_header(&s->auth_state, tag, p);

        } else if (!av_strcasecmp (tag, "Proxy-Authenticate")) {

            ff_http_auth_handle_header(&s->proxy_auth_state, tag, p);

        } else if (!av_strcasecmp (tag, "Connection")) {

            if (!strcmp(p, "close"))

                s->willclose = 1;

        } else if (!av_strcasecmp (tag, "Server") && !av_strcasecmp (p, "AkamaiGHost")) {

            s->is_akamai = 1;

        } else if (!av_strcasecmp (tag, "Content-Type") && p) {

            av_free(s->mime_type); s->mime_type = av_strdup(p);

        }

    }

    return 1;

}
