static int qcrypto_cipher_init_des_rfb(QCryptoCipher *cipher,

                                       const uint8_t *key, size_t nkey,

                                       Error **errp)

{

    QCryptoCipherBuiltin *ctxt;



    if (cipher->mode != QCRYPTO_CIPHER_MODE_ECB) {

        error_setg(errp, "Unsupported cipher mode %d", cipher->mode);

        return -1;

    }



    ctxt = g_new0(QCryptoCipherBuiltin, 1);



    ctxt->state.desrfb.key = g_new0(uint8_t, nkey);

    memcpy(ctxt->state.desrfb.key, key, nkey);

    ctxt->state.desrfb.nkey = nkey;




    ctxt->free = qcrypto_cipher_free_des_rfb;

    ctxt->setiv = qcrypto_cipher_setiv_des_rfb;

    ctxt->encrypt = qcrypto_cipher_encrypt_des_rfb;

    ctxt->decrypt = qcrypto_cipher_decrypt_des_rfb;



    cipher->opaque = ctxt;



    return 0;

}