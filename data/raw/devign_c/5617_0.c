static int rtp_open(URLContext *h, const char *uri, int flags)

{

    RTPContext *s;

    int port, is_output, ttl, local_port;

    char hostname[256];

    char buf[1024];

    char path[1024];

    const char *p;



    is_output = (flags & URL_WRONLY);



    s = av_mallocz(sizeof(RTPContext));

    if (!s)

        return AVERROR(ENOMEM);

    h->priv_data = s;



    url_split(NULL, 0, NULL, 0, hostname, sizeof(hostname), &port,

              path, sizeof(path), uri);

    /* extract parameters */

    ttl = -1;

    local_port = -1;

    p = strchr(uri, '?');

    if (p) {

        if (find_info_tag(buf, sizeof(buf), "ttl", p)) {

            ttl = strtol(buf, NULL, 10);

        }

        if (find_info_tag(buf, sizeof(buf), "localport", p)) {

            local_port = strtol(buf, NULL, 10);

        }

    }



    build_udp_url(buf, sizeof(buf),

                  hostname, port, local_port, ttl);

    if (url_open(&s->rtp_hd, buf, flags) < 0)

        goto fail;

    local_port = udp_get_local_port(s->rtp_hd);

    /* XXX: need to open another connection if the port is not even */



    /* well, should suppress localport in path */



    build_udp_url(buf, sizeof(buf),

                  hostname, port + 1, local_port + 1, ttl);

    if (url_open(&s->rtcp_hd, buf, flags) < 0)

        goto fail;



    /* just to ease handle access. XXX: need to suppress direct handle

       access */

    s->rtp_fd = udp_get_file_handle(s->rtp_hd);

    s->rtcp_fd = udp_get_file_handle(s->rtcp_hd);



    h->max_packet_size = url_get_max_packet_size(s->rtp_hd);

    h->is_streamed = 1;

    return 0;



 fail:

    if (s->rtp_hd)

        url_close(s->rtp_hd);

    if (s->rtcp_hd)

        url_close(s->rtcp_hd);

    av_free(s);

    return AVERROR(EIO);

}
