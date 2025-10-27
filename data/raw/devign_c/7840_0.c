static void gdb_accept(void *opaque)

{

    GDBState *s;

    struct sockaddr_in sockaddr;

    socklen_t len;

    int val, fd;



    for(;;) {

        len = sizeof(sockaddr);

        fd = accept(gdbserver_fd, (struct sockaddr *)&sockaddr, &len);

        if (fd < 0 && errno != EINTR) {

            perror("accept");

            return;

        } else if (fd >= 0) {

            break;

        }

    }



    /* set short latency */

    val = 1;

    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (char *)&val, sizeof(val));



    s = &gdbserver_state;

    memset (s, 0, sizeof (GDBState));

    s->env = first_cpu; /* XXX: allow to change CPU */

    s->fd = fd;

    gdb_has_xml = 0;



    gdb_syscall_state = s;



    fcntl(fd, F_SETFL, O_NONBLOCK);

}
