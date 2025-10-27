static int nbd_co_writev_1(NbdClientSession *client, int64_t sector_num,

                           int nb_sectors, QEMUIOVector *qiov,

                           int offset)

{

    struct nbd_request request;

    struct nbd_reply reply;

    ssize_t ret;



    request.type = NBD_CMD_WRITE;

    if (!bdrv_enable_write_cache(client->bs) &&

        (client->nbdflags & NBD_FLAG_SEND_FUA)) {

        request.type |= NBD_CMD_FLAG_FUA;

    }



    request.from = sector_num * 512;

    request.len = nb_sectors * 512;



    nbd_coroutine_start(client, &request);

    ret = nbd_co_send_request(client, &request, qiov, offset);

    if (ret < 0) {

        reply.error = -ret;

    } else {

        nbd_co_receive_reply(client, &request, &reply, NULL, 0);

    }

    nbd_coroutine_end(client, &request);

    return -reply.error;

}
