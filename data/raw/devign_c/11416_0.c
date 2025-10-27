int avio_check(const char *url, int flags)

{

    URLContext *h;

    int ret = ffurl_alloc(&h, url, flags, NULL);

    if (ret)

        return ret;



    if (h->prot->url_check) {

        ret = h->prot->url_check(h, flags);

    } else {

        ret = ffurl_connect(h, NULL);

        if (ret >= 0)

            ret = flags;

    }



    ffurl_close(h);

    return ret;

}
