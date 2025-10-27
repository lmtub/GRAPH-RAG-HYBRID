static void nbd_accept(void *opaque)

{

    int server_fd = (uintptr_t) opaque;

    struct sockaddr_in addr;

    socklen_t addr_len = sizeof(addr);



    int fd = accept(server_fd, (struct sockaddr *)&addr, &addr_len);

    nbd_started = true;

    if (fd >= 0 && nbd_client_new(exp, fd, nbd_client_closed)) {

        nb_fds++;

    }

}
