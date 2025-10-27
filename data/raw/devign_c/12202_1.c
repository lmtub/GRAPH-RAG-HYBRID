static int tls_read(URLContext *h, uint8_t *buf, int size)

{

    TLSContext *c = h->priv_data;

    size_t processed = 0;

    int ret = SSLRead(c->ssl_context, buf, size, &processed);

    ret = map_ssl_error(ret, processed);

    if (ret > 0)

        return ret;

    if (ret == 0)

        return AVERROR_EOF;

    return print_tls_error(h, ret);

}
