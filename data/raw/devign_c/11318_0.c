int qcrypto_pbkdf2(QCryptoHashAlgorithm hash,

                   const uint8_t *key, size_t nkey,

                   const uint8_t *salt, size_t nsalt,

                   unsigned int iterations,

                   uint8_t *out, size_t nout,

                   Error **errp)

{

    static const int hash_map[QCRYPTO_HASH_ALG__MAX] = {

        [QCRYPTO_HASH_ALG_MD5] = GCRY_MD_MD5,

        [QCRYPTO_HASH_ALG_SHA1] = GCRY_MD_SHA1,

        [QCRYPTO_HASH_ALG_SHA256] = GCRY_MD_SHA256,

    };

    int ret;



    if (hash >= G_N_ELEMENTS(hash_map) ||

        hash_map[hash] == GCRY_MD_NONE) {

        error_setg(errp, "Unexpected hash algorithm %d", hash);

        return -1;

    }



    ret = gcry_kdf_derive(key, nkey, GCRY_KDF_PBKDF2,

                          hash_map[hash],

                          salt, nsalt, iterations,

                          nout, out);

    if (ret != 0) {

        error_setg(errp, "Cannot derive password: %s",

                   gcry_strerror(ret));

        return -1;

    }



    return 0;

}
