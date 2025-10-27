void qcrypto_cipher_free(QCryptoCipher *cipher)

{

    QCryptoCipherBuiltin *ctxt = cipher->opaque;

    if (!cipher) {

        return;

    }



    ctxt->free(cipher);

    g_free(cipher);

}
