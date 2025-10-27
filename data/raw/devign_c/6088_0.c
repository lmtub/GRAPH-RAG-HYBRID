int qio_channel_socket_listen_sync(QIOChannelSocket *ioc,

                                   SocketAddress *addr,

                                   Error **errp)

{

    int fd;



    trace_qio_channel_socket_listen_sync(ioc, addr);

    fd = socket_listen(addr, errp);

    if (fd < 0) {

        trace_qio_channel_socket_listen_fail(ioc);

        return -1;

    }



    trace_qio_channel_socket_listen_complete(ioc, fd);

    if (qio_channel_socket_set_fd(ioc, fd, errp) < 0) {

        close(fd);

        return -1;

    }

    qio_channel_set_feature(QIO_CHANNEL(ioc), QIO_CHANNEL_FEATURE_LISTEN);



    return 0;

}
