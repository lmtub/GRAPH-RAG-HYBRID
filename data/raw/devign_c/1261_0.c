static ssize_t block_crypto_write_func(QCryptoBlock *block,

                                       size_t offset,

                                       const uint8_t *buf,

                                       size_t buflen,

                                       Error **errp,

                                       void *opaque)

{

    struct BlockCryptoCreateData *data = opaque;

    ssize_t ret;



    ret = blk_pwrite(data->blk, offset, buf, buflen, 0);

    if (ret < 0) {

        error_setg_errno(errp, -ret, "Could not write encryption header");

        return ret;

    }

    return ret;

}
