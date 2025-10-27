int nbd_client_co_preadv(BlockDriverState *bs, uint64_t offset,

                         uint64_t bytes, QEMUIOVector *qiov, int flags)

{

    NBDRequest request = {

        .type = NBD_CMD_READ,

        .from = offset,

        .len = bytes,

    };



    assert(bytes <= NBD_MAX_BUFFER_SIZE);

    assert(!flags);



    return nbd_co_request(bs, &request, qiov);

}
