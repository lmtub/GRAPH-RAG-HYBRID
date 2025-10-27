socket_sockaddr_to_address_unix(struct sockaddr_storage *sa,

                                socklen_t salen,

                                Error **errp)

{

    SocketAddress *addr;

    struct sockaddr_un *su = (struct sockaddr_un *)sa;



    addr = g_new0(SocketAddress, 1);

    addr->type = SOCKET_ADDRESS_KIND_UNIX;

    addr->u.q_unix = g_new0(UnixSocketAddress, 1);

    if (su->sun_path[0]) {

        addr->u.q_unix->path = g_strndup(su->sun_path,

                                         sizeof(su->sun_path));

    }



    return addr;

}
