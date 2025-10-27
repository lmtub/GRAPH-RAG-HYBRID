int tcp_start_incoming_migration(const char *host_port)

{

    struct sockaddr_in addr;

    int val;

    int s;



    if (parse_host_port(&addr, host_port) < 0) {

        fprintf(stderr, "invalid host/port combination: %s\n", host_port);

        return -EINVAL;

    }



    s = socket(PF_INET, SOCK_STREAM, 0);

    if (s == -1)

        return -socket_error();



    val = 1;

    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, (const char *)&val, sizeof(val));



    if (bind(s, (struct sockaddr *)&addr, sizeof(addr)) == -1)

        goto err;



    if (listen(s, 1) == -1)

        goto err;



    qemu_set_fd_handler2(s, NULL, tcp_accept_incoming_migration, NULL,

                         (void *)(unsigned long)s);



    return 0;



err:

    close(s);

    return -socket_error();

}
