int ffurl_get_short_seek(URLContext *h)

{

    if (!h->prot->url_get_short_seek)

        return AVERROR(ENOSYS);

    return h->prot->url_get_short_seek(h);

}
