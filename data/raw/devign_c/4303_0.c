int rtp_set_remote_url(URLContext *h, const char *uri)

{

    RTPContext *s = h->priv_data;

    char hostname[256];

    int port;



    char buf[1024];

    char path[1024];

    

    url_split(NULL, 0, hostname, sizeof(hostname), &port, 

              path, sizeof(path), uri);



    snprintf(buf, sizeof(buf), "udp://%s:%d%s", hostname, port, path);

    udp_set_remote_url(s->rtp_hd, buf);



    snprintf(buf, sizeof(buf), "udp://%s:%d%s", hostname, port + 1, path);

    udp_set_remote_url(s->rtcp_hd, buf);

    return 0;

}
