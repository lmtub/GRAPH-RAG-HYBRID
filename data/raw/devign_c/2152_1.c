static ssize_t socket_writev_buffer(void *opaque, struct iovec *iov, int iovcnt,

                                    int64_t pos)

{

    QEMUFileSocket *s = opaque;

    ssize_t len;

    ssize_t size = iov_size(iov, iovcnt);



    len = iov_send(s->fd, iov, iovcnt, 0, size);

    if (len < size) {

        len = -socket_error();

    }

    return len;

}
