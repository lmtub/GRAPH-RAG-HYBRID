static int http_connect(URLContext *h, const char *path, const char *local_path,

                        const char *hoststr, const char *auth,

                        const char *proxyauth, int *new_location)

{

    HTTPContext *s = h->priv_data;

    int post, err;

    char headers[1024] = "";

    char *authstr = NULL, *proxyauthstr = NULL;

    int64_t off = s->off;

    int len = 0;

    const char *method;





    /* send http header */

    post = h->flags & AVIO_FLAG_WRITE;



    if (s->post_data) {

        /* force POST method and disable chunked encoding when

         * custom HTTP post data is set */

        post = 1;

        s->chunked_post = 0;

    }



    method = post ? "POST" : "GET";

    authstr = ff_http_auth_create_response(&s->auth_state, auth, local_path,

                                           method);

    proxyauthstr = ff_http_auth_create_response(&s->proxy_auth_state, proxyauth,

                                                local_path, method);



    /* set default headers if needed */

    if (!has_header(s->headers, "\r\nUser-Agent: "))

       len += av_strlcatf(headers + len, sizeof(headers) - len,

                          "User-Agent: %s\r\n", LIBAVFORMAT_IDENT);

    if (!has_header(s->headers, "\r\nAccept: "))

        len += av_strlcpy(headers + len, "Accept: */*\r\n",

                          sizeof(headers) - len);

    if (!has_header(s->headers, "\r\nRange: ") && !post)

        len += av_strlcatf(headers + len, sizeof(headers) - len,

                           "Range: bytes=%"PRId64"-\r\n", s->off);



    if (!has_header(s->headers, "\r\nConnection: ")) {

        if (s->multiple_requests) {

            len += av_strlcpy(headers + len, "Connection: keep-alive\r\n",

                              sizeof(headers) - len);

        } else {

            len += av_strlcpy(headers + len, "Connection: close\r\n",

                              sizeof(headers) - len);

        }

    }



    if (!has_header(s->headers, "\r\nHost: "))

        len += av_strlcatf(headers + len, sizeof(headers) - len,

                           "Host: %s\r\n", hoststr);

    if (!has_header(s->headers, "\r\nContent-Length: ") && s->post_data)

        len += av_strlcatf(headers + len, sizeof(headers) - len,

                           "Content-Length: %d\r\n", s->post_datalen);



    /* now add in custom headers */

    if (s->headers)

        av_strlcpy(headers + len, s->headers, sizeof(headers) - len);



    snprintf(s->buffer, sizeof(s->buffer),

             "%s %s HTTP/1.1\r\n"

             "%s"

             "%s"

             "%s"

             "%s%s"

             "\r\n",

             method,

             path,

             post && s->chunked_post ? "Transfer-Encoding: chunked\r\n" : "",

             headers,

             authstr ? authstr : "",

             proxyauthstr ? "Proxy-" : "", proxyauthstr ? proxyauthstr : "");



    av_freep(&authstr);

    av_freep(&proxyauthstr);

    if ((err = ffurl_write(s->hd, s->buffer, strlen(s->buffer))) < 0)

        return err;



    if (s->post_data)

        if ((err = ffurl_write(s->hd, s->post_data, s->post_datalen)) < 0)

            return err;



    /* init input buffer */

    s->buf_ptr = s->buffer;

    s->buf_end = s->buffer;

    s->line_count = 0;

    s->off = 0;

    s->filesize = -1;

    s->willclose = 0;

    s->end_chunked_post = 0;

    s->end_header = 0;

    if (post && !s->post_data) {

        /* Pretend that it did work. We didn't read any header yet, since

         * we've still to send the POST data, but the code calling this

         * function will check http_code after we return. */

        s->http_code = 200;

        return 0;

    }



    /* wait for header */

    err = http_read_header(h, new_location);

    if (err < 0)

        return err;



    return (off == s->off) ? 0 : -1;

}
