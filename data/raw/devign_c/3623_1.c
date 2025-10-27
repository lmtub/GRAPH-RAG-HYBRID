void qemu_put_byte(QEMUFile *f, int v)

{

    if (f->last_error) {

        return;

    }



    f->buf[f->buf_index] = v;

    f->bytes_xfer++;

    if (f->ops->writev_buffer) {

        add_to_iovec(f, f->buf + f->buf_index, 1);

    }

    f->buf_index++;

    if (f->buf_index == IO_BUF_SIZE) {

        qemu_fflush(f);

    }

}
