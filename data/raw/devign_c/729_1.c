void tcp_start_incoming_migration(const char *host_port, Error **errp)

{

    int s;



    s = inet_listen(host_port, NULL, 256, SOCK_STREAM, 0, errp);

    if (s < 0) {

        return;

    }



    qemu_set_fd_handler2(s, NULL, tcp_accept_incoming_migration, NULL,

                         (void *)(intptr_t)s);

}
