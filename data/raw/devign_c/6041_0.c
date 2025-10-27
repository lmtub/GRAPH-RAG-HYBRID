qcrypto_block_luks_load_key(QCryptoBlock *block,

                            QCryptoBlockLUKSKeySlot *slot,

                            const char *password,

                            QCryptoCipherAlgorithm cipheralg,

                            QCryptoCipherMode ciphermode,

                            QCryptoHashAlgorithm hash,

                            QCryptoIVGenAlgorithm ivalg,

                            QCryptoCipherAlgorithm ivcipheralg,

                            QCryptoHashAlgorithm ivhash,

                            uint8_t *masterkey,

                            size_t masterkeylen,

                            QCryptoBlockReadFunc readfunc,

                            void *opaque,

                            Error **errp)

{

    QCryptoBlockLUKS *luks = block->opaque;

    uint8_t *splitkey;

    size_t splitkeylen;

    uint8_t *possiblekey;

    int ret = -1;

    ssize_t rv;

    QCryptoCipher *cipher = NULL;

    uint8_t keydigest[QCRYPTO_BLOCK_LUKS_DIGEST_LEN];

    QCryptoIVGen *ivgen = NULL;

    size_t niv;



    if (slot->active != QCRYPTO_BLOCK_LUKS_KEY_SLOT_ENABLED) {

        return 0;

    }



    splitkeylen = masterkeylen * slot->stripes;

    splitkey = g_new0(uint8_t, splitkeylen);

    possiblekey = g_new0(uint8_t, masterkeylen);



    /*

     * The user password is used to generate a (possible)

     * decryption key. This may or may not successfully

     * decrypt the master key - we just blindly assume

     * the key is correct and validate the results of

     * decryption later.

     */

    if (qcrypto_pbkdf2(hash,

                       (const uint8_t *)password, strlen(password),

                       slot->salt, QCRYPTO_BLOCK_LUKS_SALT_LEN,

                       slot->iterations,

                       possiblekey, masterkeylen,

                       errp) < 0) {

        goto cleanup;

    }



    /*

     * We need to read the master key material from the

     * LUKS key material header. What we're reading is

     * not the raw master key, but rather the data after

     * it has been passed through AFSplit and the result

     * then encrypted.

     */

    rv = readfunc(block,

                  slot->key_offset * QCRYPTO_BLOCK_LUKS_SECTOR_SIZE,

                  splitkey, splitkeylen,

                  errp,

                  opaque);

    if (rv < 0) {

        goto cleanup;

    }





    /* Setup the cipher/ivgen that we'll use to try to decrypt

     * the split master key material */

    cipher = qcrypto_cipher_new(cipheralg, ciphermode,

                                possiblekey, masterkeylen,

                                errp);

    if (!cipher) {

        goto cleanup;

    }



    niv = qcrypto_cipher_get_iv_len(cipheralg,

                                    ciphermode);

    ivgen = qcrypto_ivgen_new(ivalg,

                              ivcipheralg,

                              ivhash,

                              possiblekey, masterkeylen,

                              errp);

    if (!ivgen) {

        goto cleanup;

    }





    /*

     * The master key needs to be decrypted in the same

     * way that the block device payload will be decrypted

     * later. In particular we'll be using the IV generator

     * to reset the encryption cipher every time the master

     * key crosses a sector boundary.

     */

    if (qcrypto_block_decrypt_helper(cipher,

                                     niv,

                                     ivgen,

                                     QCRYPTO_BLOCK_LUKS_SECTOR_SIZE,

                                     0,

                                     splitkey,

                                     splitkeylen,

                                     errp) < 0) {

        goto cleanup;

    }



    /*

     * Now we've decrypted the split master key, join

     * it back together to get the actual master key.

     */

    if (qcrypto_afsplit_decode(hash,

                               masterkeylen,

                               slot->stripes,

                               splitkey,

                               masterkey,

                               errp) < 0) {

        goto cleanup;

    }





    /*

     * We still don't know that the masterkey we got is valid,

     * because we just blindly assumed the user's password

     * was correct. This is where we now verify it. We are

     * creating a hash of the master key using PBKDF and

     * then comparing that to the hash stored in the key slot

     * header

     */

    if (qcrypto_pbkdf2(hash,

                       masterkey, masterkeylen,

                       luks->header.master_key_salt,

                       QCRYPTO_BLOCK_LUKS_SALT_LEN,

                       luks->header.master_key_iterations,

                       keydigest, G_N_ELEMENTS(keydigest),

                       errp) < 0) {

        goto cleanup;

    }



    if (memcmp(keydigest, luks->header.master_key_digest,

               QCRYPTO_BLOCK_LUKS_DIGEST_LEN) == 0) {

        /* Success, we got the right master key */

        ret = 1;

        goto cleanup;

    }



    /* Fail, user's password was not valid for this key slot,

     * tell caller to try another slot */

    ret = 0;



 cleanup:

    qcrypto_ivgen_free(ivgen);

    qcrypto_cipher_free(cipher);

    g_free(splitkey);

    g_free(possiblekey);

    return ret;

}
