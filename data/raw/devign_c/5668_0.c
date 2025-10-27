void qemu_put_buffer(QEMUFile *f, const uint8_t *buf, int size)

{

    int l;

    while (size > 0) {

        l = IO_BUF_SIZE - f->buf_index;

        if (l > size)

            l = size;

        memcpy(f->buf + f->buf_index, buf, l);

        f->buf_index += l;

        buf += l;

        size -= l;

        if (f->buf_index >= IO_BUF_SIZE)

            qemu_fflush(f);

    }

}
