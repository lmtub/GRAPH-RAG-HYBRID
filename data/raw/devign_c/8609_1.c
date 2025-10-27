qcrypto_tls_creds_x509_unload(QCryptoTLSCredsX509 *creds)

{

    if (creds->data) {

        gnutls_certificate_free_credentials(creds->data);

        creds->data = NULL;





