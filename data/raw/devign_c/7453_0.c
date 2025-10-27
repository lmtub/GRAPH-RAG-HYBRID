static void qemu_chr_parse_udp(QemuOpts *opts, ChardevBackend *backend,

                               Error **errp)

{

    const char *host = qemu_opt_get(opts, "host");

    const char *port = qemu_opt_get(opts, "port");

    const char *localaddr = qemu_opt_get(opts, "localaddr");

    const char *localport = qemu_opt_get(opts, "localport");

    bool has_local = false;

    SocketAddress *addr;



    if (host == NULL || strlen(host) == 0) {

        host = "localhost";

    }

    if (port == NULL || strlen(port) == 0) {

        error_setg(errp, "chardev: udp: remote port not specified");

        return;

    }

    if (localport == NULL || strlen(localport) == 0) {

        localport = "0";

    } else {

        has_local = true;

    }

    if (localaddr == NULL || strlen(localaddr) == 0) {

        localaddr = "";

    } else {

        has_local = true;

    }



    backend->udp = g_new0(ChardevUdp, 1);



    addr = g_new0(SocketAddress, 1);

    addr->kind = SOCKET_ADDRESS_KIND_INET;

    addr->inet = g_new0(InetSocketAddress, 1);

    addr->inet->host = g_strdup(host);

    addr->inet->port = g_strdup(port);

    addr->inet->has_ipv4 = qemu_opt_get(opts, "ipv4");

    addr->inet->ipv4 = qemu_opt_get_bool(opts, "ipv4", 0);

    addr->inet->has_ipv6 = qemu_opt_get(opts, "ipv6");

    addr->inet->ipv6 = qemu_opt_get_bool(opts, "ipv6", 0);

    backend->udp->remote = addr;



    if (has_local) {

        backend->udp->has_local = true;

        addr = g_new0(SocketAddress, 1);

        addr->kind = SOCKET_ADDRESS_KIND_INET;

        addr->inet = g_new0(InetSocketAddress, 1);

        addr->inet->host = g_strdup(localaddr);

        addr->inet->port = g_strdup(localport);

        backend->udp->local = addr;

    }

}
