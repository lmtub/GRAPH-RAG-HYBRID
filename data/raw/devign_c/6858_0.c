static QIOChannelSocket *nbd_establish_connection(SocketAddress *saddr_flat,

                                                  Error **errp)

{

    SocketAddressLegacy *saddr = socket_address_crumple(saddr_flat);

    QIOChannelSocket *sioc;

    Error *local_err = NULL;



    sioc = qio_channel_socket_new();

    qio_channel_set_name(QIO_CHANNEL(sioc), "nbd-client");



    qio_channel_socket_connect_sync(sioc,

                                    saddr,

                                    &local_err);

    qapi_free_SocketAddressLegacy(saddr);

    if (local_err) {

        object_unref(OBJECT(sioc));

        error_propagate(errp, local_err);

        return NULL;

    }



    qio_channel_set_delay(QIO_CHANNEL(sioc), false);



    return sioc;

}
