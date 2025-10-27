int socket_dgram(SocketAddress *remote, SocketAddress *local, Error **errp)

{

    QemuOpts *opts;

    int fd;



    opts = qemu_opts_create_nofail(&socket_optslist);

    switch (remote->kind) {

    case SOCKET_ADDRESS_KIND_INET:

        qemu_opt_set(opts, "host", remote->inet->host);

        qemu_opt_set(opts, "port", remote->inet->port);

        if (local) {

            qemu_opt_set(opts, "localaddr", local->inet->host);

            qemu_opt_set(opts, "localport", local->inet->port);

        }

        fd = inet_dgram_opts(opts, errp);

        break;



    default:

        error_setg(errp, "socket type unsupported for datagram");

        return -1;

    }

    qemu_opts_del(opts);

    return fd;

}
