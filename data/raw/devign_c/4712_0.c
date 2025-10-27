static int vnc_display_connect(VncDisplay *vd,

                               SocketAddress **saddr,

                               size_t nsaddr,

                               SocketAddress **wsaddr,

                               size_t nwsaddr,

                               Error **errp)

{

    /* connect to viewer */

    QIOChannelSocket *sioc = NULL;

    if (nwsaddr != 0) {

        error_setg(errp, "Cannot use websockets in reverse mode");

        return -1;

    }

    if (nsaddr != 1) {

        error_setg(errp, "Expected a single address in reverse mode");

        return -1;

    }

    /* TODO SOCKET_ADDRESS_KIND_FD when fd has AF_UNIX */

    vd->is_unix = saddr[0]->type == SOCKET_ADDRESS_KIND_UNIX;

    sioc = qio_channel_socket_new();

    qio_channel_set_name(QIO_CHANNEL(sioc), "vnc-reverse");

    if (qio_channel_socket_connect_sync(sioc, saddr[0], errp) < 0) {

        return -1;

    }

    vnc_connect(vd, sioc, false, false);

    object_unref(OBJECT(sioc));

    return 0;

}
