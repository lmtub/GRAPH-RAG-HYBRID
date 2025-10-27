static int http_connect(URLContext *h, const char *path, const char *hoststr,

                        const char *auth, int *new_location)

{

    HTTPContext *s = h->priv_data;

    int post, err;

    char line[1024];

    char headers[1024] = "";

    char *authstr = NULL;

    int64_t off = s->off;

    int len = 0;





    /* send http header */

    post = h->flags & URL_WRONLY;

    authstr = ff_http_auth_create_response(&s->auth_state, auth, path,

                                        post ? "POST" : "GET");



    /* set default headers if needed */

    if (!has_header(s->headers, "\r\nUser-Agent: "))

       len += av_strlcatf(headers + len, sizeof(headers) - len,

                          "User-Agent: %s\r\n", LIBAVFORMAT_IDENT);

    if (!has_header(s->headers, "\r\nAccept: "))

        len += av_strlcpy(headers + len, "Accept: */*\r\n",

                          sizeof(headers) - len);

    if (!has_header(s->headers, "\r\nRange: "))

        len += av_strlcatf(headers + len, sizeof(headers) - len,

                           "Range: bytes=%"PRId64"-\r\n", s->off);

    if (!has_header(s->headers, "\r\nConnection: "))

        len += av_strlcpy(headers + len, "Connection: close\r\n",

                          sizeof(headers)-len);

    if (!has_header(s->headers, "\r\nHost: "))

        len += av_strlcatf(headers + len, sizeof(headers) - len,

                           "Host: %s\r\n", hoststr);



    /* now add in custom headers */

    av_strlcpy(headers+len, s->headers, sizeof(headers)-len);



    snprintf(s->buffer, sizeof(s->buffer),

             "%s %s HTTP/1.1\r\n"

             "%s"

             "%s"

             "%s"

             "\r\n",

             post ? "POST" : "GET",

             path,

             post && s->is_chunked ? "Transfer-Encoding: chunked\r\n" : "",

             headers,

             authstr ? authstr : "");



    av_freep(&authstr);

    if (http_write(h, s->buffer, strlen(s->buffer)) < 0)

        return AVERROR(EIO);



    /* init input buffer */

    s->buf_ptr = s->buffer;

    s->buf_end = s->buffer;

    s->line_count = 0;

    s->off = 0;

    s->filesize = -1;

    if (post) {

        /* always use chunked encoding for upload data */

        s->chunksize = 0;

        /* Pretend that it did work. We didn't read any header yet, since

         * we've still to send the POST data, but the code calling this

         * function will check http_code after we return. */

        s->http_code = 200;

        return 0;

    }



    /* wait for header */

    for(;;) {

        if (http_get_line(s, line, sizeof(line)) < 0)

            return AVERROR(EIO);



        dprintf(NULL, "header='%s'\n", line);



        err = process_line(h, line, s->line_count, new_location);

        if (err < 0)

            return err;

        if (err == 0)

            break;

        s->line_count++;

    }



    return (off == s->off) ? 0 : -1;

}
