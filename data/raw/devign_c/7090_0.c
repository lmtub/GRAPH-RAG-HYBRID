int nbd_client_session_co_flush(NbdClientSession *client)

{

    struct nbd_request request = { .type = NBD_CMD_FLUSH };

    struct nbd_reply reply;

    ssize_t ret;



    if (!(client->nbdflags & NBD_FLAG_SEND_FLUSH)) {

        return 0;

    }



    if (client->nbdflags & NBD_FLAG_SEND_FUA) {

        request.type |= NBD_CMD_FLAG_FUA;

    }



    request.from = 0;

    request.len = 0;



    nbd_coroutine_start(client, &request);

    ret = nbd_co_send_request(client, &request, NULL, 0);

    if (ret < 0) {

        reply.error = -ret;

    } else {

        nbd_co_receive_reply(client, &request, &reply, NULL, 0);

    }

    nbd_coroutine_end(client, &request);

    return -reply.error;

}
