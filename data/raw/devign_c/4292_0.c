int qcrypto_hash_bytesv(QCryptoHashAlgorithm alg,

                        const struct iovec *iov G_GNUC_UNUSED,

                        size_t niov G_GNUC_UNUSED,

                        uint8_t **result G_GNUC_UNUSED,

                        size_t *resultlen G_GNUC_UNUSED,

                        Error **errp)

{

    error_setg(errp,

               "Hash algorithm %d not supported without GNUTLS",

               alg);

    return -1;

}
