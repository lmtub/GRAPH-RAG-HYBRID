static int rtp_new_av_stream(HTTPContext *c,

                             int stream_index, struct sockaddr_in *dest_addr,

                             HTTPContext *rtsp_c)

{

    AVFormatContext *ctx;

    AVStream *st;

    char *ipaddr;

    URLContext *h = NULL;

    uint8_t *dummy_buf;

    int max_packet_size;



    /* now we can open the relevant output stream */

    ctx = avformat_alloc_context();

    if (!ctx)

        return -1;

    ctx->oformat = av_guess_format("rtp", NULL, NULL);



    st = av_mallocz(sizeof(AVStream));

    if (!st)

        goto fail;

    ctx->nb_streams = 1;

    ctx->streams[0] = st;



    if (!c->stream->feed ||

        c->stream->feed == c->stream)

        memcpy(st, c->stream->streams[stream_index], sizeof(AVStream));

    else

        memcpy(st,

               c->stream->feed->streams[c->stream->feed_streams[stream_index]],

               sizeof(AVStream));

    st->priv_data = NULL;



    /* build destination RTP address */

    ipaddr = inet_ntoa(dest_addr->sin_addr);



    switch(c->rtp_protocol) {

    case RTSP_LOWER_TRANSPORT_UDP:

    case RTSP_LOWER_TRANSPORT_UDP_MULTICAST:

        /* RTP/UDP case */



        /* XXX: also pass as parameter to function ? */

        if (c->stream->is_multicast) {

            int ttl;

            ttl = c->stream->multicast_ttl;

            if (!ttl)

                ttl = 16;

            snprintf(ctx->filename, sizeof(ctx->filename),

                     "rtp://%s:%d?multicast=1&ttl=%d",

                     ipaddr, ntohs(dest_addr->sin_port), ttl);

        } else {

            snprintf(ctx->filename, sizeof(ctx->filename),

                     "rtp://%s:%d", ipaddr, ntohs(dest_addr->sin_port));

        }



        if (url_open(&h, ctx->filename, URL_WRONLY) < 0)

            goto fail;

        c->rtp_handles[stream_index] = h;

        max_packet_size = url_get_max_packet_size(h);

        break;

    case RTSP_LOWER_TRANSPORT_TCP:

        /* RTP/TCP case */

        c->rtsp_c = rtsp_c;

        max_packet_size = RTSP_TCP_MAX_PACKET_SIZE;

        break;

    default:

        goto fail;

    }



    http_log("%s:%d - - \"PLAY %s/streamid=%d %s\"\n",

             ipaddr, ntohs(dest_addr->sin_port),

             c->stream->filename, stream_index, c->protocol);



    /* normally, no packets should be output here, but the packet size may be checked */

    if (url_open_dyn_packet_buf(&ctx->pb, max_packet_size) < 0) {

        /* XXX: close stream */

        goto fail;

    }

    av_set_parameters(ctx, NULL);

    if (av_write_header(ctx) < 0) {

    fail:

        if (h)

            url_close(h);

        av_free(ctx);

        return -1;

    }

    avio_close_dyn_buf(ctx->pb, &dummy_buf);

    av_free(dummy_buf);



    c->rtp_ctx[stream_index] = ctx;

    return 0;

}
