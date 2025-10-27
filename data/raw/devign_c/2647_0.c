int qio_channel_socket_connect_sync(QIOChannelSocket *ioc,

                                    SocketAddressLegacy *addr,

                                    Error **errp)

{

    int fd;



    trace_qio_channel_socket_connect_sync(ioc, addr);

    fd = socket_connect(addr, NULL, NULL, errp);

    if (fd < 0) {

        trace_qio_channel_socket_connect_fail(ioc);

        return -1;

    }



    trace_qio_channel_socket_connect_complete(ioc, fd);

    if (qio_channel_socket_set_fd(ioc, fd, errp) < 0) {

        close(fd);

        return -1;

    }



    return 0;

}
