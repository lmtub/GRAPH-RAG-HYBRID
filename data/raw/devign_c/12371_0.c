static void socket_start_outgoing_migration(MigrationState *s,

                                            SocketAddress *saddr,

                                            Error **errp)

{

    QIOChannelSocket *sioc = qio_channel_socket_new();

    struct SocketConnectData *data = g_new0(struct SocketConnectData, 1);



    data->s = s;

    if (saddr->type == SOCKET_ADDRESS_KIND_INET) {

        data->hostname = g_strdup(saddr->u.inet.data->host);

    }



    qio_channel_set_name(QIO_CHANNEL(sioc), "migration-socket-outgoing");

    qio_channel_socket_connect_async(sioc,

                                     saddr,

                                     socket_outgoing_migration,

                                     data,

                                     socket_connect_data_free);

    qapi_free_SocketAddress(saddr);

}
