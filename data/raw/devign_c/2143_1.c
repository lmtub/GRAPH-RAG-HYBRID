static int qcow2_set_up_encryption(BlockDriverState *bs, const char *encryptfmt,

                                   QemuOpts *opts, Error **errp)

{

    BDRVQcow2State *s = bs->opaque;

    QCryptoBlockCreateOptions *cryptoopts = NULL;

    QCryptoBlock *crypto = NULL;

    int ret = -EINVAL;

    QDict *options, *encryptopts;



    options = qemu_opts_to_qdict(opts, NULL);

    qdict_extract_subqdict(options, &encryptopts, "encrypt.");

    QDECREF(options);



    if (!g_str_equal(encryptfmt, "aes")) {

        error_setg(errp, "Unknown encryption format '%s', expected 'aes'",

                   encryptfmt);

        ret = -EINVAL;

        goto out;

    }

    cryptoopts = block_crypto_create_opts_init(

        Q_CRYPTO_BLOCK_FORMAT_QCOW, encryptopts, errp);

    if (!cryptoopts) {

        ret = -EINVAL;

        goto out;

    }

    s->crypt_method_header = QCOW_CRYPT_AES;



    crypto = qcrypto_block_create(cryptoopts,

                                  NULL, NULL,

                                  bs, errp);

    if (!crypto) {

        ret = -EINVAL;

        goto out;

    }



    ret = qcow2_update_header(bs);

    if (ret < 0) {

        error_setg_errno(errp, -ret, "Could not write encryption header");

        goto out;

    }



 out:

    QDECREF(encryptopts);

    qcrypto_block_free(crypto);

    qapi_free_QCryptoBlockCreateOptions(cryptoopts);

    return ret;

}
