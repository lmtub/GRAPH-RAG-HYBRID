static int socket_get_buffer(void *opaque, uint8_t *buf, int64_t pos, int size)

{

    QEMUFileSocket *s = opaque;

    ssize_t len;



    do {

        len = qemu_recv(s->fd, buf, size, 0);

    } while (len == -1 && socket_error() == EINTR);



    if (len == -1)

        len = -socket_error();



    return len;

}
