static void vnc_init_basic_info(SocketAddressLegacy *addr,

                                VncBasicInfo *info,

                                Error **errp)

{

    switch (addr->type) {

    case SOCKET_ADDRESS_LEGACY_KIND_INET:

        info->host = g_strdup(addr->u.inet.data->host);

        info->service = g_strdup(addr->u.inet.data->port);

        if (addr->u.inet.data->ipv6) {

            info->family = NETWORK_ADDRESS_FAMILY_IPV6;

        } else {

            info->family = NETWORK_ADDRESS_FAMILY_IPV4;

        }

        break;



    case SOCKET_ADDRESS_LEGACY_KIND_UNIX:

        info->host = g_strdup("");

        info->service = g_strdup(addr->u.q_unix.data->path);

        info->family = NETWORK_ADDRESS_FAMILY_UNIX;

        break;



    case SOCKET_ADDRESS_LEGACY_KIND_VSOCK:

    case SOCKET_ADDRESS_LEGACY_KIND_FD:

        error_setg(errp, "Unsupported socket address type %s",

                   SocketAddressLegacyKind_lookup[addr->type]);

        break;

    default:

        abort();

    }



    return;

}
