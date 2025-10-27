SocketAddress *socket_address_flatten(SocketAddressLegacy *addr_legacy)

{

    SocketAddress *addr = g_new(SocketAddress, 1);



    if (!addr_legacy) {

        return NULL;

    }



    switch (addr_legacy->type) {

    case SOCKET_ADDRESS_LEGACY_KIND_INET:

        addr->type = SOCKET_ADDRESS_TYPE_INET;

        QAPI_CLONE_MEMBERS(InetSocketAddress, &addr->u.inet,

                           addr_legacy->u.inet.data);

        break;

    case SOCKET_ADDRESS_LEGACY_KIND_UNIX:

        addr->type = SOCKET_ADDRESS_TYPE_UNIX;

        QAPI_CLONE_MEMBERS(UnixSocketAddress, &addr->u.q_unix,

                           addr_legacy->u.q_unix.data);

        break;

    case SOCKET_ADDRESS_LEGACY_KIND_VSOCK:

        addr->type = SOCKET_ADDRESS_TYPE_VSOCK;

        QAPI_CLONE_MEMBERS(VsockSocketAddress, &addr->u.vsock,

                           addr_legacy->u.vsock.data);

        break;

    case SOCKET_ADDRESS_LEGACY_KIND_FD:

        addr->type = SOCKET_ADDRESS_TYPE_FD;

        QAPI_CLONE_MEMBERS(String, &addr->u.fd, addr_legacy->u.fd.data);

        break;

    default:

        abort();

    }



    return addr;

}
