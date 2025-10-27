qcrypto_tls_creds_x509_sanity_check(QCryptoTLSCredsX509 *creds,

                                    bool isServer,

                                    const char *cacertFile,

                                    const char *certFile,

                                    Error **errp)

{

    gnutls_x509_crt_t cert = NULL;

    gnutls_x509_crt_t cacerts[MAX_CERTS];

    size_t ncacerts = 0;

    size_t i;

    int ret = -1;



    memset(cacerts, 0, sizeof(cacerts));

    if (access(certFile, R_OK) == 0) {

        cert = qcrypto_tls_creds_load_cert(creds,

                                           certFile, isServer,

                                           errp);

        if (!cert) {

            goto cleanup;

        }

    }

    if (access(cacertFile, R_OK) == 0) {

        if (qcrypto_tls_creds_load_ca_cert_list(creds,

                                                cacertFile, cacerts,

                                                MAX_CERTS, &ncacerts,

                                                errp) < 0) {

            goto cleanup;

        }

    }



    if (cert &&

        qcrypto_tls_creds_check_cert(creds,

                                     cert, certFile, isServer,

                                     false, errp) < 0) {

        goto cleanup;

    }



    for (i = 0; i < ncacerts; i++) {

        if (qcrypto_tls_creds_check_cert(creds,

                                         cacerts[i], cacertFile,

                                         isServer, true, errp) < 0) {

            goto cleanup;

        }

    }



    if (cert && ncacerts &&

        qcrypto_tls_creds_check_cert_pair(cert, certFile, cacerts,

                                          ncacerts, cacertFile,

                                          isServer, errp) < 0) {

        goto cleanup;

    }



    ret = 0;



 cleanup:

    if (cert) {

        gnutls_x509_crt_deinit(cert);

    }

    for (i = 0; i < ncacerts; i++) {

        gnutls_x509_crt_deinit(cacerts[i]);

    }

    return ret;

}
