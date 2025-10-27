static QIOChannelSocket *nbd_establish_connection(SocketAddress *saddr,

                                                  Error **errp)

{

    QIOChannelSocket *sioc;

    Error *local_err = NULL;



    sioc = qio_channel_socket_new();

    qio_channel_set_name(QIO_CHANNEL(sioc), "nbd-client");



    qio_channel_socket_connect_sync(sioc,

                                    saddr,

                                    &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        return NULL;

    }



    qio_channel_set_delay(QIO_CHANNEL(sioc), false);



    return sioc;

}
