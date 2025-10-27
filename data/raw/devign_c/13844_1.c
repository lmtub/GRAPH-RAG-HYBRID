void qemu_put_buffer_async(QEMUFile *f, const uint8_t *buf, int size)

{

    if (!f->ops->writev_buffer) {

        qemu_put_buffer(f, buf, size);

        return;

    }



    if (f->last_error) {

        return;

    }



    f->bytes_xfer += size;

    add_to_iovec(f, buf, size);

}
