int socket_dgram(SocketAddress *remote, SocketAddress *local, Error **errp)

{

    int fd;



    switch (remote->type) {

    case SOCKET_ADDRESS_KIND_INET:

        fd = inet_dgram_saddr(remote->u.inet, local ? local->u.inet : NULL, errp);

        break;



    default:

        error_setg(errp, "socket type unsupported for datagram");

        fd = -1;

    }

    return fd;

}
