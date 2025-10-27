static void socket_start_incoming_migration(SocketAddress *saddr,

                                            Error **errp)

{

    QIOChannelSocket *listen_ioc = qio_channel_socket_new();



    qio_channel_set_name(QIO_CHANNEL(listen_ioc),

                         "migration-socket-listener");



    if (qio_channel_socket_listen_sync(listen_ioc, saddr, errp) < 0) {

        object_unref(OBJECT(listen_ioc));

        qapi_free_SocketAddress(saddr);

        return;

    }



    qio_channel_add_watch(QIO_CHANNEL(listen_ioc),

                          G_IO_IN,

                          socket_accept_incoming_migration,

                          listen_ioc,

                          (GDestroyNotify)object_unref);

    qapi_free_SocketAddress(saddr);

}
