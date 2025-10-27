static CharDriverState *qmp_chardev_open_udp(const char *id,

                                             ChardevBackend *backend,

                                             ChardevReturn *ret,

                                             Error **errp)

{

    ChardevUdp *udp = backend->u.udp;

    ChardevCommon *common = qapi_ChardevUdp_base(udp);

    QIOChannelSocket *sioc = qio_channel_socket_new();



    if (qio_channel_socket_dgram_sync(sioc,

                                      udp->local, udp->remote,

                                      errp) < 0) {

        object_unref(OBJECT(sioc));

        return NULL;

    }

    return qemu_chr_open_udp(sioc, common, errp);

}
