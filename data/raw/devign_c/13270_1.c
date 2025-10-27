void qemu_put_buffer(QEMUFile *f, const uint8_t *buf, int size)

{

    int l;



    if (f->last_error) {

        return;

    }



    while (size > 0) {

        l = IO_BUF_SIZE - f->buf_index;

        if (l > size) {

            l = size;

        }

        memcpy(f->buf + f->buf_index, buf, l);

        f->bytes_xfer += l;

        if (f->ops->writev_buffer) {

            add_to_iovec(f, f->buf + f->buf_index, l);

        }

        f->buf_index += l;

        if (f->buf_index == IO_BUF_SIZE) {

            qemu_fflush(f);

        }

        if (qemu_file_get_error(f)) {

            break;

        }

        buf += l;

        size -= l;

    }

}
