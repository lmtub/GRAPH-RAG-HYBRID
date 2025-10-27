static int tls_open(URLContext *h, const char *uri, int flags)

{

    TLSContext *c = h->priv_data;

    int ret;

    int port;

    char buf[200], host[200];

    int numerichost = 0;

    struct addrinfo hints = { 0 }, *ai = NULL;

    const char *proxy_path;

    int use_proxy;



    ff_tls_init();



    av_url_split(NULL, 0, NULL, 0, host, sizeof(host), &port, NULL, 0, uri);

    ff_url_join(buf, sizeof(buf), "tcp", NULL, host, port, NULL);



    hints.ai_flags = AI_NUMERICHOST;

    if (!getaddrinfo(host, NULL, &hints, &ai)) {

        numerichost = 1;

        freeaddrinfo(ai);

    }



    proxy_path = getenv("http_proxy");

    use_proxy = !ff_http_match_no_proxy(getenv("no_proxy"), host) &&

                proxy_path != NULL && av_strstart(proxy_path, "http://", NULL);



    if (use_proxy) {

        char proxy_host[200], proxy_auth[200], dest[200];

        int proxy_port;

        av_url_split(NULL, 0, proxy_auth, sizeof(proxy_auth),

                     proxy_host, sizeof(proxy_host), &proxy_port, NULL, 0,

                     proxy_path);

        ff_url_join(dest, sizeof(dest), NULL, NULL, host, port, NULL);

        ff_url_join(buf, sizeof(buf), "httpproxy", proxy_auth, proxy_host,

                    proxy_port, "/%s", dest);

    }



    ret = ffurl_open(&c->tcp, buf, AVIO_FLAG_READ_WRITE,

                     &h->interrupt_callback, NULL);

    if (ret)

        goto fail;

    c->fd = ffurl_get_file_handle(c->tcp);



#if CONFIG_GNUTLS

    gnutls_init(&c->session, GNUTLS_CLIENT);

    if (!numerichost)

        gnutls_server_name_set(c->session, GNUTLS_NAME_DNS, host, strlen(host));

    gnutls_certificate_allocate_credentials(&c->cred);

    gnutls_certificate_set_verify_flags(c->cred, 0);

    gnutls_credentials_set(c->session, GNUTLS_CRD_CERTIFICATE, c->cred);

    gnutls_transport_set_ptr(c->session, (gnutls_transport_ptr_t)

                                         (intptr_t) c->fd);

    gnutls_priority_set_direct(c->session, "NORMAL", NULL);

    while (1) {

        ret = gnutls_handshake(c->session);

        if (ret == 0)

            break;

        if ((ret = do_tls_poll(h, ret)) < 0)

            goto fail;

    }

#elif CONFIG_OPENSSL

    c->ctx = SSL_CTX_new(TLSv1_client_method());

    if (!c->ctx) {

        av_log(h, AV_LOG_ERROR, "%s\n", ERR_error_string(ERR_get_error(), NULL));

        ret = AVERROR(EIO);

        goto fail;

    }

    c->ssl = SSL_new(c->ctx);

    if (!c->ssl) {

        av_log(h, AV_LOG_ERROR, "%s\n", ERR_error_string(ERR_get_error(), NULL));

        ret = AVERROR(EIO);

        goto fail;

    }

    SSL_set_fd(c->ssl, c->fd);

    if (!numerichost)

        SSL_set_tlsext_host_name(c->ssl, host);

    while (1) {

        ret = SSL_connect(c->ssl);

        if (ret > 0)

            break;

        if (ret == 0) {

            av_log(h, AV_LOG_ERROR, "Unable to negotiate TLS/SSL session\n");

            ret = AVERROR(EIO);

            goto fail;

        }

        if ((ret = do_tls_poll(h, ret)) < 0)

            goto fail;

    }

#endif

    return 0;

fail:

    TLS_free(c);

    if (c->tcp)

        ffurl_close(c->tcp);

    ff_tls_deinit();

    return ret;

}
