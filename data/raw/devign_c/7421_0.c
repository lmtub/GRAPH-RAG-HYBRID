static ssize_t qio_channel_file_writev(QIOChannel *ioc,

                                       const struct iovec *iov,

                                       size_t niov,

                                       int *fds,

                                       size_t nfds,

                                       Error **errp)

{

    QIOChannelFile *fioc = QIO_CHANNEL_FILE(ioc);

    ssize_t ret;



 retry:

    ret = writev(fioc->fd, iov, niov);

    if (ret <= 0) {

        if (errno == EAGAIN ||

            errno == EWOULDBLOCK) {

            return QIO_CHANNEL_ERR_BLOCK;

        }

        if (errno == EINTR) {

            goto retry;

        }

        error_setg_errno(errp, errno,

                         "Unable to write to file");

        return -1;

    }

    return ret;

}
