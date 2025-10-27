void nbd_client_close(BlockDriverState *bs)

{

    NbdClientSession *client = nbd_get_client_session(bs);

    struct nbd_request request = {

        .type = NBD_CMD_DISC,

        .from = 0,

        .len = 0

    };



    if (client->ioc == NULL) {

        return;

    }



    nbd_send_request(client->ioc, &request);



    nbd_teardown_connection(bs);

}
