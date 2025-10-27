int ffurl_shutdown(URLContext *h, int flags)

{

    if (!h->prot->url_shutdown)

        return AVERROR(EINVAL);

    return h->prot->url_shutdown(h, flags);

}
