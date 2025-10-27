size_t qcrypto_cipher_get_block_len(QCryptoCipherAlgorithm alg)

{

    if (alg >= G_N_ELEMENTS(alg_key_len)) {

        return 0;

    }

    return alg_block_len[alg];

}
