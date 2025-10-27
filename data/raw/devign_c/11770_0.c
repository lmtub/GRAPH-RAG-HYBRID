static ssize_t qio_channel_socket_readv(QIOChannel *ioc,

                                        const struct iovec *iov,

                                        size_t niov,

                                        int **fds,

                                        size_t *nfds,

                                        Error **errp)

{

    QIOChannelSocket *sioc = QIO_CHANNEL_SOCKET(ioc);

    ssize_t ret;

    struct msghdr msg = { NULL, };

    char control[CMSG_SPACE(sizeof(int) * SOCKET_MAX_FDS)];

    int sflags = 0;



    memset(control, 0, CMSG_SPACE(sizeof(int) * SOCKET_MAX_FDS));



#ifdef MSG_CMSG_CLOEXEC

    sflags |= MSG_CMSG_CLOEXEC;

#endif



    msg.msg_iov = (struct iovec *)iov;

    msg.msg_iovlen = niov;

    if (fds && nfds) {

        msg.msg_control = control;

        msg.msg_controllen = sizeof(control);

    }



 retry:

    ret = recvmsg(sioc->fd, &msg, sflags);

    if (ret < 0) {

        if (socket_error() == EAGAIN ||

            socket_error() == EWOULDBLOCK) {

            return QIO_CHANNEL_ERR_BLOCK;

        }

        if (socket_error() == EINTR) {

            goto retry;

        }



        error_setg_errno(errp, socket_error(),

                         "Unable to read from socket");

        return -1;

    }



    if (fds && nfds) {

        qio_channel_socket_copy_fds(&msg, fds, nfds);

    }



    return ret;

}
