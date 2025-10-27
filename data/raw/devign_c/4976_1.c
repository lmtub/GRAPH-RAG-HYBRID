int qemu_peek_byte(QEMUFile *f, int offset)

{

    int index = f->buf_index + offset;



    assert(!qemu_file_is_writable(f));

    assert(offset < IO_BUF_SIZE);



    if (index >= f->buf_size) {

        qemu_fill_buffer(f);

        index = f->buf_index + offset;

        if (index >= f->buf_size) {

            return 0;

        }

    }

    return f->buf[index];

}
