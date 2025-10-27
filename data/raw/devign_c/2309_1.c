void fd_start_outgoing_migration(MigrationState *s, const char *fdname, Error **errp)

{

    int fd = monitor_get_fd(cur_mon, fdname, errp);

    if (fd == -1) {

        return;

    }

    s->file = qemu_fdopen(fd, "wb");



    migrate_fd_connect(s);

}
