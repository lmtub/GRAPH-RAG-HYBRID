int qcrypto_cipher_decrypt(QCryptoCipher *cipher,
                           const void *in,
                           void *out,
                           size_t len,
                           Error **errp)
{
    QCryptoCipherBuiltin *ctxt = cipher->opaque;
    return ctxt->decrypt(cipher, in, out, len, errp);