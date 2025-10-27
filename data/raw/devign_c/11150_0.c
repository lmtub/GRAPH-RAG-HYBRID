int qcrypto_hash_bytesv(QCryptoHashAlgorithm alg,

                        const struct iovec *iov,

                        size_t niov,

                        uint8_t **result,

                        size_t *resultlen,

                        Error **errp)

{

    int i, ret;

    gnutls_hash_hd_t dig;



    if (alg >= G_N_ELEMENTS(qcrypto_hash_alg_map)) {

        error_setg(errp,

                   "Unknown hash algorithm %d",

                   alg);

        return -1;

    }



    ret = gnutls_hash_init(&dig, qcrypto_hash_alg_map[alg]);



    if (ret < 0) {

        error_setg(errp,

                   "Unable to initialize hash algorithm: %s",

                   gnutls_strerror(ret));

        return -1;

    }



    for (i = 0; i < niov; i++) {

        ret = gnutls_hash(dig, iov[i].iov_base, iov[i].iov_len);

        if (ret < 0) {

            error_setg(errp,

                       "Unable process hash data: %s",

                       gnutls_strerror(ret));

            goto error;

        }

    }



    ret = gnutls_hash_get_len(qcrypto_hash_alg_map[alg]);

    if (ret <= 0) {

        error_setg(errp,

                   "Unable to get hash length: %s",

                   gnutls_strerror(ret));

        goto error;

    }

    if (*resultlen == 0) {

        *resultlen = ret;

        *result = g_new0(uint8_t, *resultlen);

    } else if (*resultlen != ret) {

        error_setg(errp,

                   "Result buffer size %zu is smaller than hash %d",

                   *resultlen, ret);

        goto error;

    }



    gnutls_hash_deinit(dig, *result);

    return 0;



 error:

    gnutls_hash_deinit(dig, NULL);

    return -1;

}
