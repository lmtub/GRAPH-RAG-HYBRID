static int nbd_co_readv_1(BlockDriverState *bs, int64_t sector_num,

                          int nb_sectors, QEMUIOVector *qiov,

                          int offset)

{

    BDRVNBDState *s = bs->opaque;

    struct nbd_request request;

    struct nbd_reply reply;



    request.type = NBD_CMD_READ;

    request.from = sector_num * 512;

    request.len = nb_sectors * 512;



    nbd_coroutine_start(s, &request);

    if (nbd_co_send_request(s, &request, NULL, 0) == -1) {

        reply.error = errno;

    } else {

        nbd_co_receive_reply(s, &request, &reply, qiov->iov, offset);

    }

    nbd_coroutine_end(s, &request);

    return -reply.error;



}
