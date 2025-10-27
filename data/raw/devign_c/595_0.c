int tcp_start_outgoing_migration(MigrationState *s, const char *host_port,

                                 Error **errp)

{

    s->get_error = socket_errno;

    s->write = socket_write;

    s->close = tcp_close;



    s->fd = inet_connect(host_port, false, errp);



    if (!error_is_set(errp)) {

        migrate_fd_connect(s);

    } else if (error_is_type(*errp, QERR_SOCKET_CONNECT_IN_PROGRESS)) {

        DPRINTF("connect in progress\n");

        qemu_set_fd_handler2(s->fd, NULL, NULL, tcp_wait_for_connect, s);

    } else if (error_is_type(*errp, QERR_SOCKET_CREATE_FAILED)) {

        DPRINTF("connect failed\n");

        return -1;

    } else if (error_is_type(*errp, QERR_SOCKET_CONNECT_FAILED)) {

        DPRINTF("connect failed\n");

        migrate_fd_error(s);

        return -1;

    } else {

        DPRINTF("unknown error\n");

        return -1;

    }



    return 0;

}
