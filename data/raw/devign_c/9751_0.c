static void migrate_fd_put_notify(void *opaque)

{

    MigrationState *s = opaque;



    qemu_set_fd_handler2(s->fd, NULL, NULL, NULL, NULL);

    qemu_file_put_notify(s->file);

    if (qemu_file_get_error(s->file)) {

        migrate_fd_error(s);

    }

}
