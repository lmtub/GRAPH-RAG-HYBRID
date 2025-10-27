static void unix_wait_for_connect(int fd, Error *err, void *opaque)

{

    MigrationState *s = opaque;



    if (fd < 0) {

        DPRINTF("migrate connect error: %s\n", error_get_pretty(err));

        s->file = NULL;

        migrate_fd_error(s);

    } else {

        DPRINTF("migrate connect success\n");

        s->file = qemu_fopen_socket(fd, "wb");

        migrate_fd_connect(s);

    }

}
