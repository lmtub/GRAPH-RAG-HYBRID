qcrypto_tls_session_new(QCryptoTLSCreds *creds,

                        const char *hostname,

                        const char *aclname,

                        QCryptoTLSCredsEndpoint endpoint,

                        Error **errp)

{

    QCryptoTLSSession *session;

    int ret;



    session = g_new0(QCryptoTLSSession, 1);

    trace_qcrypto_tls_session_new(

        session, creds, hostname ? hostname : "<none>",

        aclname ? aclname : "<none>", endpoint);



    if (hostname) {

        session->hostname = g_strdup(hostname);

    }

    if (aclname) {

        session->aclname = g_strdup(aclname);

    }

    session->creds = creds;

    object_ref(OBJECT(creds));



    if (creds->endpoint != endpoint) {

        error_setg(errp, "Credentials endpoint doesn't match session");

        goto error;

    }



    if (endpoint == QCRYPTO_TLS_CREDS_ENDPOINT_SERVER) {

        ret = gnutls_init(&session->handle, GNUTLS_SERVER);

    } else {

        ret = gnutls_init(&session->handle, GNUTLS_CLIENT);

    }

    if (ret < 0) {

        error_setg(errp, "Cannot initialize TLS session: %s",

                   gnutls_strerror(ret));

        goto error;

    }



    if (object_dynamic_cast(OBJECT(creds),

                            TYPE_QCRYPTO_TLS_CREDS_ANON)) {

        QCryptoTLSCredsAnon *acreds = QCRYPTO_TLS_CREDS_ANON(creds);



        ret = gnutls_priority_set_direct(session->handle,

                                         "NORMAL:+ANON-DH", NULL);

        if (ret < 0) {

            error_setg(errp, "Unable to set TLS session priority: %s",

                       gnutls_strerror(ret));

            goto error;

        }

        if (creds->endpoint == QCRYPTO_TLS_CREDS_ENDPOINT_SERVER) {

            ret = gnutls_credentials_set(session->handle,

                                         GNUTLS_CRD_ANON,

                                         acreds->data.server);

        } else {

            ret = gnutls_credentials_set(session->handle,

                                         GNUTLS_CRD_ANON,

                                         acreds->data.client);

        }

        if (ret < 0) {

            error_setg(errp, "Cannot set session credentials: %s",

                       gnutls_strerror(ret));

            goto error;

        }

    } else if (object_dynamic_cast(OBJECT(creds),

                                   TYPE_QCRYPTO_TLS_CREDS_X509)) {

        QCryptoTLSCredsX509 *tcreds = QCRYPTO_TLS_CREDS_X509(creds);



        ret = gnutls_set_default_priority(session->handle);

        if (ret < 0) {

            error_setg(errp, "Cannot set default TLS session priority: %s",

                       gnutls_strerror(ret));

            goto error;

        }

        ret = gnutls_credentials_set(session->handle,

                                     GNUTLS_CRD_CERTIFICATE,

                                     tcreds->data);

        if (ret < 0) {

            error_setg(errp, "Cannot set session credentials: %s",

                       gnutls_strerror(ret));

            goto error;

        }



        if (creds->endpoint == QCRYPTO_TLS_CREDS_ENDPOINT_SERVER) {

            /* This requests, but does not enforce a client cert.

             * The cert checking code later does enforcement */

            gnutls_certificate_server_set_request(session->handle,

                                                  GNUTLS_CERT_REQUEST);

        }

    } else {

        error_setg(errp, "Unsupported TLS credentials type %s",

                   object_get_typename(OBJECT(creds)));

        goto error;

    }



    gnutls_transport_set_ptr(session->handle, session);

    gnutls_transport_set_push_function(session->handle,

                                       qcrypto_tls_session_push);

    gnutls_transport_set_pull_function(session->handle,

                                       qcrypto_tls_session_pull);



    return session;



 error:

    qcrypto_tls_session_free(session);

    return NULL;

}
