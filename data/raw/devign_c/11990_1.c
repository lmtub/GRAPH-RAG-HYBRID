int ff_http_do_new_request(URLContext *h, const char *uri)
{
    HTTPContext *s = h->priv_data;
    AVDictionary *options = NULL;
    int ret;
    ret = http_shutdown(h, h->flags);
    if (ret < 0)
        return ret;
    s->end_chunked_post = 0;
    s->chunkend      = 0;
    s->off           = 0;
    s->icy_data_read = 0;
    av_free(s->location);
    s->location = av_strdup(uri);
    if (!s->location)
        return AVERROR(ENOMEM);
    ret = http_open_cnx(h, &options);
    av_dict_free(&options);
    return ret;