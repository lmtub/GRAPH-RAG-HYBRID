static void nbd_teardown_connection(NbdClientSession *client)

{

    struct nbd_request request = {

        .type = NBD_CMD_DISC,

        .from = 0,

        .len = 0

    };



    nbd_send_request(client->sock, &request);



    /* finish any pending coroutines */

    shutdown(client->sock, 2);

    nbd_recv_coroutines_enter_all(client);



    qemu_aio_set_fd_handler(client->sock, NULL, NULL, NULL);

    closesocket(client->sock);

    client->sock = -1;

}
