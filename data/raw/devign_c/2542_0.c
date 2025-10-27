static void nbd_teardown_connection(NbdClientSession *client)

{

    /* finish any pending coroutines */

    shutdown(client->sock, 2);

    nbd_recv_coroutines_enter_all(client);



    nbd_client_session_detach_aio_context(client);

    closesocket(client->sock);

    client->sock = -1;

}
