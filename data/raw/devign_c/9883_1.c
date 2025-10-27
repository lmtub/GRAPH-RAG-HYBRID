qio_channel_socket_accept(QIOChannelSocket *ioc,

                          Error **errp)

{

    QIOChannelSocket *cioc;



    cioc = qio_channel_socket_new();

    cioc->remoteAddrLen = sizeof(ioc->remoteAddr);

    cioc->localAddrLen = sizeof(ioc->localAddr);



 retry:

    trace_qio_channel_socket_accept(ioc);

    cioc->fd = qemu_accept(ioc->fd, (struct sockaddr *)&cioc->remoteAddr,

                           &cioc->remoteAddrLen);

    if (cioc->fd < 0) {

        trace_qio_channel_socket_accept_fail(ioc);

        if (errno == EINTR) {

            goto retry;

        }

        goto error;

    }



    if (getsockname(cioc->fd, (struct sockaddr *)&cioc->localAddr,

                    &cioc->localAddrLen) < 0) {

        error_setg_errno(errp, errno,

                         "Unable to query local socket address");

        goto error;

    }



#ifndef WIN32

    if (cioc->localAddr.ss_family == AF_UNIX) {

        QIOChannel *ioc_local = QIO_CHANNEL(cioc);

        qio_channel_set_feature(ioc_local, QIO_CHANNEL_FEATURE_FD_PASS);

    }

#endif /* WIN32 */



    trace_qio_channel_socket_accept_complete(ioc, cioc, cioc->fd);

    return cioc;



 error:

    object_unref(OBJECT(cioc));

    return NULL;

}
