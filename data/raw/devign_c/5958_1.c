static gnutls_certificate_credentials_t vnc_tls_initialize_x509_cred(VncDisplay *vd)

{

    gnutls_certificate_credentials_t x509_cred;

    int ret;



    if (!vd->tls.x509cacert) {

        VNC_DEBUG("No CA x509 certificate specified\n");

        return NULL;

    }

    if (!vd->tls.x509cert) {

        VNC_DEBUG("No server x509 certificate specified\n");

        return NULL;

    }

    if (!vd->tls.x509key) {

        VNC_DEBUG("No server private key specified\n");

        return NULL;

    }



    if ((ret = gnutls_certificate_allocate_credentials(&x509_cred)) < 0) {

        VNC_DEBUG("Cannot allocate credentials %s\n", gnutls_strerror(ret));

        return NULL;

    }

    if ((ret = gnutls_certificate_set_x509_trust_file(x509_cred,

                                                      vd->tls.x509cacert,

                                                      GNUTLS_X509_FMT_PEM)) < 0) {

        VNC_DEBUG("Cannot load CA certificate %s\n", gnutls_strerror(ret));

        gnutls_certificate_free_credentials(x509_cred);

        return NULL;

    }



    if ((ret = gnutls_certificate_set_x509_key_file (x509_cred,

                                                     vd->tls.x509cert,

                                                     vd->tls.x509key,

                                                     GNUTLS_X509_FMT_PEM)) < 0) {

        VNC_DEBUG("Cannot load certificate & key %s\n", gnutls_strerror(ret));

        gnutls_certificate_free_credentials(x509_cred);

        return NULL;

    }



    if (vd->tls.x509cacrl) {

        if ((ret = gnutls_certificate_set_x509_crl_file(x509_cred,

                                                        vd->tls.x509cacrl,

                                                        GNUTLS_X509_FMT_PEM)) < 0) {

            VNC_DEBUG("Cannot load CRL %s\n", gnutls_strerror(ret));

            gnutls_certificate_free_credentials(x509_cred);

            return NULL;

        }

    }



    gnutls_certificate_set_dh_params (x509_cred, dh_params);



    return x509_cred;

}
