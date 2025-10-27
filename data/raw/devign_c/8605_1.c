static int unix_close(MigrationState *s)

{

    DPRINTF("unix_close\n");

    if (s->fd != -1) {

        close(s->fd);

        s->fd = -1;

    }

    return 0;

}
