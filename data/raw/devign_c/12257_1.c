static int nbd_co_readv_1(NbdClientSession *client, int64_t sector_num,

                          int nb_sectors, QEMUIOVector *qiov,

                          int offset)

{

    struct nbd_request request;

    struct nbd_reply reply;

    ssize_t ret;



    request.type = NBD_CMD_READ;

    request.from = sector_num * 512;

    request.len = nb_sectors * 512;



    nbd_coroutine_start(client, &request);

    ret = nbd_co_send_request(client, &request, NULL, 0);

    if (ret < 0) {

        reply.error = -ret;

    } else {

        nbd_co_receive_reply(client, &request, &reply, qiov, offset);

    }

    nbd_coroutine_end(client, &request);

    return -reply.error;



}
