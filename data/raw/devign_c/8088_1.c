static int fd_close(MigrationState *s)

{

    DPRINTF("fd_close\n");

    if (s->fd != -1) {

        close(s->fd);

        s->fd = -1;

    }

    return 0;

}
