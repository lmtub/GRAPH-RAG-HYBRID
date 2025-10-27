int qio_channel_socket_dgram_sync(QIOChannelSocket *ioc,

                                  SocketAddressLegacy *localAddr,

                                  SocketAddressLegacy *remoteAddr,

                                  Error **errp)

{

    int fd;



    trace_qio_channel_socket_dgram_sync(ioc, localAddr, remoteAddr);

    fd = socket_dgram(remoteAddr, localAddr, errp);

    if (fd < 0) {

        trace_qio_channel_socket_dgram_fail(ioc);

        return -1;

    }



    trace_qio_channel_socket_dgram_complete(ioc, fd);

    if (qio_channel_socket_set_fd(ioc, fd, errp) < 0) {

        close(fd);

        return -1;

    }



    return 0;

}
