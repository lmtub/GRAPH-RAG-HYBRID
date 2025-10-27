static SocketAddressLegacy *unix_build_address(const char *path)

{

    SocketAddressLegacy *saddr;



    saddr = g_new0(SocketAddressLegacy, 1);

    saddr->type = SOCKET_ADDRESS_LEGACY_KIND_UNIX;

    saddr->u.q_unix.data = g_new0(UnixSocketAddress, 1);

    saddr->u.q_unix.data->path = g_strdup(path);



    return saddr;

}
