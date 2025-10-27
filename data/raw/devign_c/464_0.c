static int qcow_set_key(BlockDriverState *bs, const char *key)

{

    BDRVQcowState *s = bs->opaque;

    uint8_t keybuf[16];

    int len, i;

    Error *err;



    memset(keybuf, 0, 16);

    len = strlen(key);

    if (len > 16)

        len = 16;

    /* XXX: we could compress the chars to 7 bits to increase

       entropy */

    for(i = 0;i < len;i++) {

        keybuf[i] = key[i];

    }

    assert(bs->encrypted);



    qcrypto_cipher_free(s->cipher);

    s->cipher = qcrypto_cipher_new(

        QCRYPTO_CIPHER_ALG_AES_128,

        QCRYPTO_CIPHER_MODE_CBC,

        keybuf, G_N_ELEMENTS(keybuf),

        &err);



    if (!s->cipher) {

        /* XXX would be nice if errors in this method could

         * be properly propagate to the caller. Would need

         * the bdrv_set_key() API signature to be fixed. */

        error_free(err);

        return -1;

    }

    return 0;

}
