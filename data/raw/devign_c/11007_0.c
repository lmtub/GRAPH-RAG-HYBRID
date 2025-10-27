gboolean qcrypto_hash_supports(QCryptoHashAlgorithm alg)

{

    if (alg < G_N_ELEMENTS(qcrypto_hash_alg_map)) {

        return true;

    }

    return false;

}
