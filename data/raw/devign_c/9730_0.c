static void net_socket_accept(void *opaque)

{

    NetSocketListenState *s = opaque;

    NetSocketState *s1;

    struct sockaddr_in saddr;

    socklen_t len;

    int fd;



    for(;;) {

        len = sizeof(saddr);

        fd = qemu_accept(s->fd, (struct sockaddr *)&saddr, &len);

        if (fd < 0 && errno != EINTR) {

            return;

        } else if (fd >= 0) {

            break;

        }

    }

    s1 = net_socket_fd_init(s->peer, s->model, s->name, fd, 1);

    if (s1) {

        snprintf(s1->nc.info_str, sizeof(s1->nc.info_str),

                 "socket: connection from %s:%d",

                 inet_ntoa(saddr.sin_addr), ntohs(saddr.sin_port));

    }

}
