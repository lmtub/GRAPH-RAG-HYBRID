static void socket_start_outgoing_migration(MigrationState *s,

                                            SocketAddress *saddr,

                                            Error **errp)

{

    QIOChannelSocket *sioc = qio_channel_socket_new();

    qio_channel_socket_connect_async(sioc,

                                     saddr,

                                     socket_outgoing_migration,

                                     s,

                                     NULL);

    qapi_free_SocketAddress(saddr);

}
