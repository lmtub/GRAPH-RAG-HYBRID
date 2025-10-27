static void nbd_client_close(NBDClient *client)

{

    qemu_set_fd_handler2(client->sock, NULL, NULL, NULL, NULL);

    close(client->sock);

    client->sock = -1;

    if (client->close) {

        client->close(client);

    }

    nbd_client_put(client);

}
