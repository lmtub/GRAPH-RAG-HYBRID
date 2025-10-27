static void pty_chr_read(void *opaque)

{

    CharDriverState *chr = opaque;

    PtyCharDriver *s = chr->opaque;

    int size, len;

    uint8_t buf[1024];



    len = sizeof(buf);

    if (len > s->read_bytes)

        len = s->read_bytes;

    if (len == 0)

        return;

    size = read(s->fd, buf, len);

    if ((size == -1 && errno == EIO) ||

        (size == 0)) {

        pty_chr_state(chr, 0);

        return;

    }

    if (size > 0) {

        pty_chr_state(chr, 1);

        qemu_chr_read(chr, buf, size);

    }

}
