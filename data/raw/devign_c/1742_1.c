int vnc_tls_set_x509_creds_dir(VncDisplay *vd,

                               const char *certdir)

{

    if (vnc_set_x509_credential(vd, certdir, X509_CA_CERT_FILE, &vd->tls.x509cacert, 0) < 0)

        goto cleanup;

    if (vnc_set_x509_credential(vd, certdir, X509_CA_CRL_FILE, &vd->tls.x509cacrl, 1) < 0)

        goto cleanup;

    if (vnc_set_x509_credential(vd, certdir, X509_SERVER_CERT_FILE, &vd->tls.x509cert, 0) < 0)

        goto cleanup;

    if (vnc_set_x509_credential(vd, certdir, X509_SERVER_KEY_FILE, &vd->tls.x509key, 0) < 0)

        goto cleanup;



    return 0;



 cleanup:

    g_free(vd->tls.x509cacert);

    g_free(vd->tls.x509cacrl);

    g_free(vd->tls.x509cert);

    g_free(vd->tls.x509key);

    vd->tls.x509cacert = vd->tls.x509cacrl = vd->tls.x509cert = vd->tls.x509key = NULL;

    return -1;

}
