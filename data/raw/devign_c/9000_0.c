static int unix_connect_saddr(UnixSocketAddress *saddr,

                              NonBlockingConnectHandler *callback, void *opaque,

                              Error **errp)

{

    struct sockaddr_un un;

    ConnectState *connect_state = NULL;

    int sock, rc;



    if (saddr->path == NULL) {

        error_setg(errp, "unix connect: no path specified");

        return -1;

    }



    sock = qemu_socket(PF_UNIX, SOCK_STREAM, 0);

    if (sock < 0) {

        error_setg_errno(errp, errno, "Failed to create socket");

        return -1;

    }

    if (callback != NULL) {

        connect_state = g_malloc0(sizeof(*connect_state));

        connect_state->callback = callback;

        connect_state->opaque = opaque;

        qemu_set_nonblock(sock);

    }



    memset(&un, 0, sizeof(un));

    un.sun_family = AF_UNIX;

    snprintf(un.sun_path, sizeof(un.sun_path), "%s", saddr->path);



    /* connect to peer */

    do {

        rc = 0;

        if (connect(sock, (struct sockaddr *) &un, sizeof(un)) < 0) {

            rc = -errno;

        }

    } while (rc == -EINTR);



    if (connect_state != NULL && QEMU_SOCKET_RC_INPROGRESS(rc)) {

        connect_state->fd = sock;

        qemu_set_fd_handler(sock, NULL, wait_for_connect, connect_state);

        return sock;

    } else if (rc >= 0) {

        /* non blocking socket immediate success, call callback */

        if (callback != NULL) {

            callback(sock, NULL, opaque);

        }

    }



    if (rc < 0) {

        error_setg_errno(errp, -rc, "Failed to connect socket");

        close(sock);

        sock = -1;

    }



    g_free(connect_state);

    return sock;

}
