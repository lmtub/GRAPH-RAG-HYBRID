static ssize_t qcow2_crypto_hdr_init_func(QCryptoBlock *block, size_t headerlen,

                                          void *opaque, Error **errp)

{

    BlockDriverState *bs = opaque;

    BDRVQcow2State *s = bs->opaque;

    int64_t ret;

    int64_t clusterlen;



    ret = qcow2_alloc_clusters(bs, headerlen);

    if (ret < 0) {

        error_setg_errno(errp, -ret,

                         "Cannot allocate cluster for LUKS header size %zu",

                         headerlen);

        return -1;

    }



    s->crypto_header.length = headerlen;

    s->crypto_header.offset = ret;



    /* Zero fill remaining space in cluster so it has predictable

     * content in case of future spec changes */

    clusterlen = size_to_clusters(s, headerlen) * s->cluster_size;


    ret = bdrv_pwrite_zeroes(bs->file,

                             ret + headerlen,

                             clusterlen - headerlen, 0);

    if (ret < 0) {

        error_setg_errno(errp, -ret, "Could not zero fill encryption header");

        return -1;

    }



    return ret;

}