static ssize_t block_crypto_init_func(QCryptoBlock *block,

                                      void *opaque,

                                      size_t headerlen,

                                      Error **errp)

{

    struct BlockCryptoCreateData *data = opaque;

    int ret;



    /* User provided size should reflect amount of space made

     * available to the guest, so we must take account of that

     * which will be used by the crypto header

     */

    data->size += headerlen;



    qemu_opt_set_number(data->opts, BLOCK_OPT_SIZE, data->size, &error_abort);

    ret = bdrv_create_file(data->filename, data->opts, errp);

    if (ret < 0) {

        return -1;

    }



    data->blk = blk_new_open(data->filename, NULL, NULL,

                             BDRV_O_RDWR | BDRV_O_PROTOCOL, errp);

    if (!data->blk) {

        return -1;

    }



    return 0;

}
