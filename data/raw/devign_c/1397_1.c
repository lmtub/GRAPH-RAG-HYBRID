static void socket_outgoing_migration(Object *src,

                                      Error *err,

                                      gpointer opaque)

{

    MigrationState *s = opaque;

    QIOChannel *sioc = QIO_CHANNEL(src);



    if (err) {

        trace_migration_socket_outgoing_error(error_get_pretty(err));

        s->to_dst_file = NULL;

        migrate_fd_error(s, err);

    } else {

        trace_migration_socket_outgoing_connected();

        migration_set_outgoing_channel(s, sioc);

    }

    object_unref(src);

}
