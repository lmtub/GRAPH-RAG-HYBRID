void qemu_fopen_ops_buffered(MigrationState *migration_state)

{

    QEMUFileBuffered *s;



    s = g_malloc0(sizeof(*s));



    s->migration_state = migration_state;

    s->xfer_limit = migration_state->bandwidth_limit / 10;

    s->migration_state->complete = false;



    s->file = qemu_fopen_ops(s, &buffered_file_ops);



    migration_state->file = s->file;



    qemu_thread_create(&s->thread, buffered_file_thread, s,

                       QEMU_THREAD_DETACHED);

}
