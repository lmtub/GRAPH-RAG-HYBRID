int net_init_socket(const Netdev *netdev, const char *name,

                    NetClientState *peer, Error **errp)

{

    /* FIXME error_setg(errp, ...) on failure */

    Error *err = NULL;

    const NetdevSocketOptions *sock;



    assert(netdev->type == NET_CLIENT_DRIVER_SOCKET);

    sock = &netdev->u.socket;



    if (sock->has_fd + sock->has_listen + sock->has_connect + sock->has_mcast +

        sock->has_udp != 1) {

        error_report("exactly one of fd=, listen=, connect=, mcast= or udp="

                     " is required");

        return -1;

    }



    if (sock->has_localaddr && !sock->has_mcast && !sock->has_udp) {

        error_report("localaddr= is only valid with mcast= or udp=");

        return -1;

    }



    if (sock->has_fd) {

        int fd;



        fd = monitor_fd_param(cur_mon, sock->fd, &err);

        if (fd == -1) {

            error_report_err(err);

            return -1;

        }

        qemu_set_nonblock(fd);

        if (!net_socket_fd_init(peer, "socket", name, fd, 1)) {

            return -1;

        }

        return 0;

    }



    if (sock->has_listen) {

        if (net_socket_listen_init(peer, "socket", name, sock->listen) == -1) {

            return -1;

        }

        return 0;

    }



    if (sock->has_connect) {

        if (net_socket_connect_init(peer, "socket", name, sock->connect) ==

            -1) {

            return -1;

        }

        return 0;

    }



    if (sock->has_mcast) {

        /* if sock->localaddr is missing, it has been initialized to "all bits

         * zero" */

        if (net_socket_mcast_init(peer, "socket", name, sock->mcast,

            sock->localaddr) == -1) {

            return -1;

        }

        return 0;

    }



    assert(sock->has_udp);

    if (!sock->has_localaddr) {

        error_report("localaddr= is mandatory with udp=");

        return -1;

    }

    if (net_socket_udp_init(peer, "socket", name, sock->udp, sock->localaddr) ==

        -1) {

        return -1;

    }

    return 0;

}
