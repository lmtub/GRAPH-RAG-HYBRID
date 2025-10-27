void migrate_fd_connect(MigrationState *s)

{

    s->state = MIG_STATE_ACTIVE;

    trace_migrate_set_state(MIG_STATE_ACTIVE);



    s->bytes_xfer = 0;

    /* This is a best 1st approximation. ns to ms */

    s->expected_downtime = max_downtime/1000000;



    s->cleanup_bh = qemu_bh_new(migrate_fd_cleanup, s);

    s->file = qemu_fopen_ops(s, &migration_file_ops);



    qemu_file_set_rate_limit(s->file,

                             s->bandwidth_limit / XFER_LIMIT_RATIO);



    qemu_thread_create(&s->thread, migration_thread, s,

                       QEMU_THREAD_JOINABLE);

    notifier_list_notify(&migration_state_notifiers, s);

}
