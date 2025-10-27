qcrypto_tls_creds_x509_load(QCryptoTLSCredsX509 *creds,
                            Error **errp)
{
    char *cacert = NULL, *cacrl = NULL, *cert = NULL,
        *key = NULL, *dhparams = NULL;
    int ret;
    int rv = -1;
    trace_qcrypto_tls_creds_x509_load(creds,
            creds->parent_obj.dir ? creds->parent_obj.dir : "<nodir>");
    if (creds->parent_obj.endpoint == QCRYPTO_TLS_CREDS_ENDPOINT_SERVER) {
        if (qcrypto_tls_creds_get_path(&creds->parent_obj,
                                       QCRYPTO_TLS_CREDS_X509_CA_CERT,
                                       true, &cacert, errp) < 0 ||
            qcrypto_tls_creds_get_path(&creds->parent_obj,
                                       QCRYPTO_TLS_CREDS_X509_CA_CRL,
                                       false, &cacrl, errp) < 0 ||
            qcrypto_tls_creds_get_path(&creds->parent_obj,
                                       QCRYPTO_TLS_CREDS_X509_SERVER_CERT,
                                       true, &cert, errp) < 0 ||
            qcrypto_tls_creds_get_path(&creds->parent_obj,
                                       QCRYPTO_TLS_CREDS_X509_SERVER_KEY,
                                       true, &key, errp) < 0 ||
            qcrypto_tls_creds_get_path(&creds->parent_obj,
                                       QCRYPTO_TLS_CREDS_DH_PARAMS,
                                       false, &dhparams, errp) < 0) {
    } else {
        if (qcrypto_tls_creds_get_path(&creds->parent_obj,
                                       QCRYPTO_TLS_CREDS_X509_CA_CERT,
                                       true, &cacert, errp) < 0 ||
            qcrypto_tls_creds_get_path(&creds->parent_obj,
                                       QCRYPTO_TLS_CREDS_X509_CLIENT_CERT,
                                       false, &cert, errp) < 0 ||
            qcrypto_tls_creds_get_path(&creds->parent_obj,
                                       QCRYPTO_TLS_CREDS_X509_CLIENT_KEY,
                                       false, &key, errp) < 0) {
    ret = gnutls_certificate_allocate_credentials(&creds->data);
    if (ret < 0) {
        error_setg(errp, "Cannot allocate credentials: '%s'",
                   gnutls_strerror(ret));
    ret = gnutls_certificate_set_x509_trust_file(creds->data,
                                                 cacert,
                                                 GNUTLS_X509_FMT_PEM);
    if (ret < 0) {
        error_setg(errp, "Cannot load CA certificate '%s': %s",
                   cacert, gnutls_strerror(ret));
    if (cert != NULL && key != NULL) {
        ret = gnutls_certificate_set_x509_key_file(creds->data,
                                                   cert, key,
                                                   GNUTLS_X509_FMT_PEM);
        if (ret < 0) {
            error_setg(errp, "Cannot load certificate '%s' & key '%s': %s",
                       cert, key, gnutls_strerror(ret));
    if (cacrl != NULL) {
        ret = gnutls_certificate_set_x509_crl_file(creds->data,
                                                   cacrl,
                                                   GNUTLS_X509_FMT_PEM);
        if (ret < 0) {
            error_setg(errp, "Cannot load CRL '%s': %s",
                       cacrl, gnutls_strerror(ret));
    if (creds->parent_obj.endpoint == QCRYPTO_TLS_CREDS_ENDPOINT_SERVER) {
        if (qcrypto_tls_creds_get_dh_params_file(&creds->parent_obj, dhparams,
                                                 &creds->parent_obj.dh_params,
                                                 errp) < 0) {
        gnutls_certificate_set_dh_params(creds->data,
                                         creds->parent_obj.dh_params);
    rv = 0;
 cleanup:
    g_free(cacert);
    g_free(cacrl);
    g_free(cert);
    g_free(key);
    g_free(dhparams);
    return rv;