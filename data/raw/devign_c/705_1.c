static int inet_connect_addr(struct addrinfo *addr, bool block,

                             bool *in_progress)

{

    int sock, rc;



    if (in_progress) {

        *in_progress = false;

    }



    sock = qemu_socket(addr->ai_family, addr->ai_socktype, addr->ai_protocol);

    if (sock < 0) {

        fprintf(stderr, "%s: socket(%s): %s\n", __func__,

                inet_strfamily(addr->ai_family), strerror(errno));

        return -1;

    }

    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));

    if (!block) {

        socket_set_nonblock(sock);

    }

    /* connect to peer */

    do {

        rc = 0;

        if (connect(sock, addr->ai_addr, addr->ai_addrlen) < 0) {

            rc = -socket_error();

        }

    } while (rc == -EINTR);



    if (!block && QEMU_SOCKET_RC_INPROGRESS(rc)) {

        if (in_progress) {

            *in_progress = true;

        }

    } else if (rc < 0) {

        closesocket(sock);

        return -1;

    }

    return sock;

}
