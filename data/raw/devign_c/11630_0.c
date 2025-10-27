static int block_crypto_create_generic(QCryptoBlockFormat format,

                                       const char *filename,

                                       QemuOpts *opts,

                                       Error **errp)

{

    int ret = -EINVAL;

    QCryptoBlockCreateOptions *create_opts = NULL;

    QCryptoBlock *crypto = NULL;

    struct BlockCryptoCreateData data = {

        .size = ROUND_UP(qemu_opt_get_size_del(opts, BLOCK_OPT_SIZE, 0),

                         BDRV_SECTOR_SIZE),

        .opts = opts,

        .filename = filename,

    };



    create_opts = block_crypto_create_opts_init(format, opts, errp);

    if (!create_opts) {

        return -1;

    }



    crypto = qcrypto_block_create(create_opts,

                                  block_crypto_init_func,

                                  block_crypto_write_func,

                                  &data,

                                  errp);



    if (!crypto) {

        ret = -EIO;

        goto cleanup;

    }



    ret = 0;

 cleanup:

    qcrypto_block_free(crypto);

    blk_unref(data.blk);

    qapi_free_QCryptoBlockCreateOptions(create_opts);

    return ret;

}
