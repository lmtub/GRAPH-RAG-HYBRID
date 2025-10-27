void exec_start_outgoing_migration(MigrationState *s, const char *command, Error **errp)

{

    QIOChannel *ioc;

    const char *argv[] = { "/bin/sh", "-c", command, NULL };



    trace_migration_exec_outgoing(command);

    ioc = QIO_CHANNEL(qio_channel_command_new_spawn(argv,

                                                    O_WRONLY,

                                                    errp));

    if (!ioc) {

        return;

    }



    migration_set_outgoing_channel(s, ioc);

    object_unref(OBJECT(ioc));

}
