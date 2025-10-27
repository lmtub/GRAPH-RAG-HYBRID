static void qemu_fill_buffer(QEMUFile *f)

{

    int len;

    int pending;



    assert(!qemu_file_is_writable(f));



    pending = f->buf_size - f->buf_index;

    if (pending > 0) {

        memmove(f->buf, f->buf + f->buf_index, pending);

    }

    f->buf_index = 0;

    f->buf_size = pending;



    len = f->ops->get_buffer(f->opaque, f->buf + pending, f->pos,

                        IO_BUF_SIZE - pending);

    if (len > 0) {

        f->buf_size += len;

        f->pos += len;

    } else if (len == 0) {

        qemu_file_set_error(f, -EIO);

    } else if (len != -EAGAIN) {

        qemu_file_set_error(f, len);

    }

}
