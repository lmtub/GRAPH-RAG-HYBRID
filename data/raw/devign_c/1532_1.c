void migration_set_incoming_channel(MigrationState *s,

                                    QIOChannel *ioc)

{

    QEMUFile *f = qemu_fopen_channel_input(ioc);



    process_incoming_migration(f);

}
