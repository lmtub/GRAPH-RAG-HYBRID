static void tap_receive(void *opaque, const uint8_t *buf, size_t size)

{

    TAPState *s = opaque;

    int ret;

    for(;;) {

        ret = write(s->fd, buf, size);

        if (ret < 0 && (errno == EINTR || errno == EAGAIN)) {

        } else {

            break;

        }

    }

}
