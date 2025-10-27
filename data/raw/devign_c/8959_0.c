static ssize_t tap_receive_iov(void *opaque, const struct iovec *iov,

                               int iovcnt)

{

    TAPState *s = opaque;

    ssize_t len;



    do {

        len = writev(s->fd, iov, iovcnt);

    } while (len == -1 && (errno == EINTR || errno == EAGAIN));



    return len;

}
