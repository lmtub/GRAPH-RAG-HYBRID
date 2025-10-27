static int nbd_co_writev_1(BlockDriverState *bs, int64_t sector_num,

                           int nb_sectors, QEMUIOVector *qiov,

                           int offset)

{

    BDRVNBDState *s = bs->opaque;

    struct nbd_request request;

    struct nbd_reply reply;



    request.type = NBD_CMD_WRITE;

    if (!bdrv_enable_write_cache(bs) && (s->nbdflags & NBD_FLAG_SEND_FUA)) {

        request.type |= NBD_CMD_FLAG_FUA;

    }



    request.from = sector_num * 512;

    request.len = nb_sectors * 512;



    nbd_coroutine_start(s, &request);

    if (nbd_co_send_request(s, &request, qiov->iov, offset) == -1) {

        reply.error = errno;

    } else {

        nbd_co_receive_reply(s, &request, &reply, NULL, 0);

    }

    nbd_coroutine_end(s, &request);

    return -reply.error;

}
