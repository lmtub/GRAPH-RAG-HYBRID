static int pty_chr_write(CharDriverState *chr, const uint8_t *buf, int len)

{

    PtyCharDriver *s = chr->opaque;



    if (!s->connected) {

        /* guest sends data, check for (re-)connect */

        pty_chr_update_read_handler_locked(chr);

        return 0;

    }

    return io_channel_send(s->fd, buf, len);

}
