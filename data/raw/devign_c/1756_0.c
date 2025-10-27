void tcp_start_incoming_migration(const char *host_port, Error **errp)

{

    Error *err = NULL;

    SocketAddressLegacy *saddr = tcp_build_address(host_port, &err);

    if (!err) {

        socket_start_incoming_migration(saddr, &err);

    }

    error_propagate(errp, err);

}
