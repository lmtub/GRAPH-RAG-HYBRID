int socket_connect(SocketAddressLegacy *addr, NonBlockingConnectHandler *callback,

                   void *opaque, Error **errp)

{

    int fd;



    switch (addr->type) {

    case SOCKET_ADDRESS_LEGACY_KIND_INET:

        fd = inet_connect_saddr(addr->u.inet.data, callback, opaque, errp);

        break;



    case SOCKET_ADDRESS_LEGACY_KIND_UNIX:

        fd = unix_connect_saddr(addr->u.q_unix.data, callback, opaque, errp);

        break;



    case SOCKET_ADDRESS_LEGACY_KIND_FD:

        fd = monitor_get_fd(cur_mon, addr->u.fd.data->str, errp);

        if (fd >= 0 && callback) {

            qemu_set_nonblock(fd);

            callback(fd, NULL, opaque);

        }

        break;



    case SOCKET_ADDRESS_LEGACY_KIND_VSOCK:

        fd = vsock_connect_saddr(addr->u.vsock.data, callback, opaque, errp);

        break;



    default:

        abort();

    }

    return fd;

}
