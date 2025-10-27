static int unix_get_buffer(void *opaque, uint8_t *buf, int64_t pos, int size)

{

    QEMUFileSocket *s = opaque;

    ssize_t len;



    for (;;) {

        len = read(s->fd, buf, size);

        if (len != -1) {

            break;

        }

        if (errno == EAGAIN) {

            yield_until_fd_readable(s->fd);

        } else if (errno != EINTR) {

            break;

        }

    }



    if (len == -1) {

        len = -errno;

    }

    return len;

}
