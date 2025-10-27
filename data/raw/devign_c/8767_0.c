static int http_open_cnx_internal(URLContext *h, AVDictionary **options)

{

    const char *path, *proxy_path, *lower_proto = "tcp", *local_path;

    char hostname[1024], hoststr[1024], proto[10];

    char auth[1024], proxyauth[1024] = "";

    char path1[MAX_URL_SIZE];

    char buf[1024], urlbuf[MAX_URL_SIZE];

    int port, use_proxy, err, location_changed = 0;

    HTTPContext *s = h->priv_data;



    av_url_split(proto, sizeof(proto), auth, sizeof(auth),

                 hostname, sizeof(hostname), &port,

                 path1, sizeof(path1), s->location);

    ff_url_join(hoststr, sizeof(hoststr), NULL, NULL, hostname, port, NULL);



    proxy_path = getenv("http_proxy");

    use_proxy  = !ff_http_match_no_proxy(getenv("no_proxy"), hostname) &&

                 proxy_path != NULL && av_strstart(proxy_path, "http://", NULL);



    if (!strcmp(proto, "https")) {

        lower_proto = "tls";

        use_proxy   = 0;

        if (port < 0)

            port = 443;

    }

    if (port < 0)

        port = 80;



    if (path1[0] == '\0')

        path = "/";

    else

        path = path1;

    local_path = path;

    if (use_proxy) {

        /* Reassemble the request URL without auth string - we don't

         * want to leak the auth to the proxy. */

        ff_url_join(urlbuf, sizeof(urlbuf), proto, NULL, hostname, port, "%s",

                    path1);

        path = urlbuf;

        av_url_split(NULL, 0, proxyauth, sizeof(proxyauth),

                     hostname, sizeof(hostname), &port, NULL, 0, proxy_path);

    }



    ff_url_join(buf, sizeof(buf), lower_proto, NULL, hostname, port, NULL);



    if (!s->hd) {

        err = ffurl_open(&s->hd, buf, AVIO_FLAG_READ_WRITE,

                         &h->interrupt_callback, options);

        if (err < 0)

            return err;

    }



    err = http_connect(h, path, local_path, hoststr,

                       auth, proxyauth, &location_changed);

    if (err < 0)

        return err;



    return location_changed;

}
