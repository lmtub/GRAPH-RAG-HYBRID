static void qmp_chardev_open_udp(Chardev *chr,

                                 ChardevBackend *backend,

                                 bool *be_opened,

                                 Error **errp)

{

    ChardevUdp *udp = backend->u.udp.data;

    QIOChannelSocket *sioc = qio_channel_socket_new();

    char *name;

    UdpChardev *s = UDP_CHARDEV(chr);



    if (qio_channel_socket_dgram_sync(sioc,

                                      udp->local, udp->remote,

                                      errp) < 0) {

        object_unref(OBJECT(sioc));

        return;

    }



    name = g_strdup_printf("chardev-udp-%s", chr->label);

    qio_channel_set_name(QIO_CHANNEL(sioc), name);

    g_free(name);



    s->ioc = QIO_CHANNEL(sioc);

    /* be isn't opened until we get a connection */

    *be_opened = false;

}
