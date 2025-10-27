char *socket_address_to_string(struct SocketAddressLegacy *addr, Error **errp)

{

    char *buf;

    InetSocketAddress *inet;



    switch (addr->type) {

    case SOCKET_ADDRESS_LEGACY_KIND_INET:

        inet = addr->u.inet.data;

        if (strchr(inet->host, ':') == NULL) {

            buf = g_strdup_printf("%s:%s", inet->host, inet->port);

        } else {

            buf = g_strdup_printf("[%s]:%s", inet->host, inet->port);

        }

        break;



    case SOCKET_ADDRESS_LEGACY_KIND_UNIX:

        buf = g_strdup(addr->u.q_unix.data->path);

        break;



    case SOCKET_ADDRESS_LEGACY_KIND_FD:

        buf = g_strdup(addr->u.fd.data->str);

        break;



    case SOCKET_ADDRESS_LEGACY_KIND_VSOCK:

        buf = g_strdup_printf("%s:%s",

                              addr->u.vsock.data->cid,

                              addr->u.vsock.data->port);

        break;



    default:

        abort();

    }

    return buf;

}
