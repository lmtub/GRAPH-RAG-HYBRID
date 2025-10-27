size_t qcrypto_hash_digest_len(QCryptoHashAlgorithm alg)

{

    if (alg >= G_N_ELEMENTS(qcrypto_hash_alg_size)) {

        return 0;

    }

    return qcrypto_hash_alg_size[alg];

}
