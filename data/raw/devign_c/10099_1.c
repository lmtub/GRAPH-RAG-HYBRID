static int http_proxy_open(URLContext *h, const char *uri, int flags)

{

    HTTPContext *s = h->priv_data;

    char hostname[1024], hoststr[1024];

    char auth[1024], pathbuf[1024], *path;

    char line[1024], lower_url[100];

    int port, ret = 0;

    HTTPAuthType cur_auth_type;

    char *authstr;



    h->is_streamed = 1;



    av_url_split(NULL, 0, auth, sizeof(auth), hostname, sizeof(hostname), &port,

                 pathbuf, sizeof(pathbuf), uri);

    ff_url_join(hoststr, sizeof(hoststr), NULL, NULL, hostname, port, NULL);

    path = pathbuf;

    if (*path == '/')

        path++;



    ff_url_join(lower_url, sizeof(lower_url), "tcp", NULL, hostname, port,

                NULL);

redo:

    ret = ffurl_open(&s->hd, lower_url, AVIO_FLAG_READ_WRITE,

                     &h->interrupt_callback, NULL);

    if (ret < 0)

        return ret;



    authstr = ff_http_auth_create_response(&s->proxy_auth_state, auth,

                                           path, "CONNECT");

    snprintf(s->buffer, sizeof(s->buffer),

             "CONNECT %s HTTP/1.1\r\n"

             "Host: %s\r\n"

             "Connection: close\r\n"

             "%s%s"

             "\r\n",

             path,

             hoststr,

             authstr ? "Proxy-" : "", authstr ? authstr : "");

    av_freep(&authstr);



    if ((ret = ffurl_write(s->hd, s->buffer, strlen(s->buffer))) < 0)

        goto fail;



    s->buf_ptr = s->buffer;

    s->buf_end = s->buffer;

    s->line_count = 0;

    s->filesize = -1;

    cur_auth_type = s->proxy_auth_state.auth_type;



    for (;;) {

        int new_loc;

        // Note: This uses buffering, potentially reading more than the

        // HTTP header. If tunneling a protocol where the server starts

        // the conversation, we might buffer part of that here, too.

        // Reading that requires using the proper ffurl_read() function

        // on this URLContext, not using the fd directly (as the tls

        // protocol does). This shouldn't be an issue for tls though,

        // since the client starts the conversation there, so there

        // is no extra data that we might buffer up here.

        if (http_get_line(s, line, sizeof(line)) < 0) {

            ret = AVERROR(EIO);

            goto fail;

        }



        av_dlog(h, "header='%s'\n", line);



        ret = process_line(h, line, s->line_count, &new_loc);

        if (ret < 0)

            goto fail;

        if (ret == 0)

            break;

        s->line_count++;

    }

    if (s->http_code == 407 && cur_auth_type == HTTP_AUTH_NONE &&

        s->proxy_auth_state.auth_type != HTTP_AUTH_NONE) {

        ffurl_close(s->hd);

        s->hd = NULL;

        goto redo;

    }



    if (s->http_code < 400)

        return 0;

    ret = AVERROR(EIO);



fail:

    http_proxy_close(h);

    return ret;

}
