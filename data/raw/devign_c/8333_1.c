static void rtsp_cmd_describe(HTTPContext *c, const char *url)

{

    FFStream *stream;

    char path1[1024];

    const char *path;

    uint8_t *content;

    int content_length, len;

    struct sockaddr_in my_addr;



    /* find which url is asked */

    url_split(NULL, 0, NULL, 0, NULL, 0, NULL, path1, sizeof(path1), url);

    path = path1;

    if (*path == '/')

        path++;



    for(stream = first_stream; stream != NULL; stream = stream->next) {

        if (!stream->is_feed && !strcmp(stream->fmt->name, "rtp") &&

            !strcmp(path, stream->filename)) {

            goto found;

        }

    }

    /* no stream found */

    rtsp_reply_error(c, RTSP_STATUS_SERVICE); /* XXX: right error ? */

    return;



 found:

    /* prepare the media description in sdp format */



    /* get the host IP */

    len = sizeof(my_addr);

    getsockname(c->fd, (struct sockaddr *)&my_addr, &len);

    content_length = prepare_sdp_description(stream, &content, my_addr.sin_addr);

    if (content_length < 0) {

        rtsp_reply_error(c, RTSP_STATUS_INTERNAL);

        return;

    }

    rtsp_reply_header(c, RTSP_STATUS_OK);

    url_fprintf(c->pb, "Content-Type: application/sdp\r\n");

    url_fprintf(c->pb, "Content-Length: %d\r\n", content_length);

    url_fprintf(c->pb, "\r\n");

    put_buffer(c->pb, content, content_length);

}
