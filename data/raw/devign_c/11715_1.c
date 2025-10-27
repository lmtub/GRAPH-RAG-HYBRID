static int tls_open(URLContext *h, const char *uri, int flags, AVDictionary **options)

{

    TLSContext *c = h->priv_data;

    TLSShared *s = &c->tls_shared;

    int ret;



    if ((ret = ff_tls_open_underlying(s, h, uri, options)) < 0)

        goto fail;



    c->ssl_context = SSLCreateContext(NULL, s->listen ? kSSLServerSide : kSSLClientSide, kSSLStreamType);

    if (!c->ssl_context) {

        av_log(h, AV_LOG_ERROR, "Unable to create SSL context\n");

        ret = AVERROR(ENOMEM);

        goto fail;

    }

    if (s->ca_file) {

        if ((ret = load_ca(h)) < 0)

            goto fail;

        CHECK_ERROR(SSLSetSessionOption, c->ssl_context, kSSLSessionOptionBreakOnServerAuth, true);

    }

    if (s->cert_file)

        if ((ret = load_cert(h)) < 0)

            goto fail;

    if (s->verify)

        CHECK_ERROR(SSLSetPeerDomainName, c->ssl_context, s->host, strlen(s->host));

    CHECK_ERROR(SSLSetIOFuncs, c->ssl_context, tls_read_cb, tls_write_cb);

    CHECK_ERROR(SSLSetConnection, c->ssl_context, h);

    while (1) {

        OSStatus status = SSLHandshake(c->ssl_context);

        if (status == errSSLServerAuthCompleted) {

            SecTrustRef peerTrust;

            SecTrustResultType trustResult;

            if (!s->verify)

                continue;



            if (SSLCopyPeerTrust(c->ssl_context, &peerTrust) != noErr) {

                ret = AVERROR(ENOMEM);

                goto fail;

            }



            if (SecTrustSetAnchorCertificates(peerTrust, c->ca_array) != noErr) {

                ret = AVERROR_UNKNOWN;

                goto fail;

            }



            if (SecTrustEvaluate(peerTrust, &trustResult) != noErr) {

                ret = AVERROR_UNKNOWN;

                goto fail;

            }



            if (trustResult == kSecTrustResultProceed ||

                trustResult == kSecTrustResultUnspecified) {

                // certificate is trusted

                status = errSSLWouldBlock; // so we call SSLHandshake again

            } else if (trustResult == kSecTrustResultRecoverableTrustFailure) {

                // not trusted, for some reason other than being expired

                status = errSSLXCertChainInvalid;

            } else {

                // cannot use this certificate (fatal)

                status = errSSLBadCert;

            }



            if (peerTrust)

                CFRelease(peerTrust);

        }

        if (status == noErr)

            break;



        av_log(h, AV_LOG_ERROR, "Unable to negotiate TLS/SSL session: %i\n", (int)status);

        ret = AVERROR(EIO);

        goto fail;

    }



    return 0;

fail:

    tls_close(h);

    return ret;

}
