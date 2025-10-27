void qmp_nbd_server_start(SocketAddress *addr,

                          bool has_tls_creds, const char *tls_creds,

                          Error **errp)

{

    if (nbd_server) {

        error_setg(errp, "NBD server already running");

        return;

    }



    nbd_server = g_new0(NBDServerData, 1);

    nbd_server->watch = -1;

    nbd_server->listen_ioc = qio_channel_socket_new();

    qio_channel_set_name(QIO_CHANNEL(nbd_server->listen_ioc),

                         "nbd-listener");

    if (qio_channel_socket_listen_sync(

            nbd_server->listen_ioc, addr, errp) < 0) {

        goto error;

    }



    if (has_tls_creds) {

        nbd_server->tlscreds = nbd_get_tls_creds(tls_creds, errp);

        if (!nbd_server->tlscreds) {

            goto error;

        }



        /* TODO SOCKET_ADDRESS_KIND_FD where fd has AF_INET or AF_INET6 */

        if (addr->type != SOCKET_ADDRESS_KIND_INET) {

            error_setg(errp, "TLS is only supported with IPv4/IPv6");

            goto error;

        }

    }



    nbd_server->watch = qio_channel_add_watch(

        QIO_CHANNEL(nbd_server->listen_ioc),

        G_IO_IN,

        nbd_accept,

        NULL,

        NULL);



    return;



 error:

    nbd_server_free(nbd_server);

    nbd_server = NULL;

}
