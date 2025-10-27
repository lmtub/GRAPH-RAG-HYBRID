static SocketAddress *sd_socket_address(const char *path,

                                        const char *host, const char *port)

{

    SocketAddress *addr = g_new0(SocketAddress, 1);



    if (path) {

        addr->type = SOCKET_ADDRESS_KIND_UNIX;

        addr->u.q_unix.data = g_new0(UnixSocketAddress, 1);

        addr->u.q_unix.data->path = g_strdup(path);

    } else {

        addr->type = SOCKET_ADDRESS_KIND_INET;

        addr->u.inet.data = g_new0(InetSocketAddress, 1);

        addr->u.inet.data->host = g_strdup(host ?: SD_DEFAULT_ADDR);

        addr->u.inet.data->port = g_strdup(port ?: stringify(SD_DEFAULT_PORT));

    }



    return addr;

}
