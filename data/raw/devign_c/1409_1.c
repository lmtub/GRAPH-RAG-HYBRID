int qemu_peek_buffer(QEMUFile *f, uint8_t *buf, int size, size_t offset)

{

    int pending;

    int index;



    assert(!qemu_file_is_writable(f));



    index = f->buf_index + offset;

    pending = f->buf_size - index;

    if (pending < size) {

        qemu_fill_buffer(f);

        index = f->buf_index + offset;

        pending = f->buf_size - index;

    }



    if (pending <= 0) {

        return 0;

    }

    if (size > pending) {

        size = pending;

    }



    memcpy(buf, f->buf + index, size);

    return size;

}
