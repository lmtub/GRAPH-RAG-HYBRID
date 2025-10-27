char *socket_address_to_string(struct SocketAddress *addr, Error **errp)

{

    char *buf;

    InetSocketAddress *inet;

    char host_port[INET6_ADDRSTRLEN + 5 + 4];



    switch (addr->type) {

    case SOCKET_ADDRESS_KIND_INET:

        inet = addr->u.inet.data;

        if (strchr(inet->host, ':') == NULL) {

            snprintf(host_port, sizeof(host_port), "%s:%s", inet->host,

                    inet->port);

            buf = g_strdup(host_port);

        } else {

            snprintf(host_port, sizeof(host_port), "[%s]:%s", inet->host,

                    inet->port);

            buf = g_strdup(host_port);

        }

        break;



    case SOCKET_ADDRESS_KIND_UNIX:

        buf = g_strdup(addr->u.q_unix.data->path);

        break;



    case SOCKET_ADDRESS_KIND_FD:

        buf = g_strdup(addr->u.fd.data->str);

        break;



    case SOCKET_ADDRESS_KIND_VSOCK:

        buf = g_strdup_printf("%s:%s",

                              addr->u.vsock.data->cid,

                              addr->u.vsock.data->port);

        break;



    default:

        error_setg(errp, "socket family %d unsupported",

                   addr->type);

        return NULL;

    }

    return buf;

}
