void unix_start_incoming_migration(const char *path, Error **errp)

{

    SocketAddressLegacy *saddr = unix_build_address(path);

    socket_start_incoming_migration(saddr, errp);

}
