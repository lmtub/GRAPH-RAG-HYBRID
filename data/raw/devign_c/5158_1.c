static void tcp_chr_read(void *opaque)

{

    CharDriverState *chr = opaque;

    TCPCharDriver *s = chr->opaque;

    uint8_t buf[READ_BUF_LEN];

    int len, size;



    if (!s->connected || s->max_size <= 0)

        return;

    len = sizeof(buf);

    if (len > s->max_size)

        len = s->max_size;

    size = tcp_chr_recv(chr, (void *)buf, len);

    if (size == 0) {

        /* connection closed */

        s->connected = 0;

        if (s->listen_fd >= 0) {

            qemu_set_fd_handler(s->listen_fd, tcp_chr_accept, NULL, chr);

        }

        qemu_set_fd_handler(s->fd, NULL, NULL, NULL);

        closesocket(s->fd);

        s->fd = -1;

        qemu_chr_event(chr, CHR_EVENT_CLOSED);

    } else if (size > 0) {

        if (s->do_telnetopt)

            tcp_chr_process_IAC_bytes(chr, s, buf, &size);

        if (size > 0)

            qemu_chr_read(chr, buf, size);

        if (s->msgfd != -1) {

            close(s->msgfd);

            s->msgfd = -1;

        }

    }

}
