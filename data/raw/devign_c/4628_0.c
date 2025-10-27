void qemu_put_byte(QEMUFile *f, int v)

{

    if (!f->last_error && f->is_write == 0 && f->buf_index > 0) {

        fprintf(stderr,

                "Attempted to write to buffer while read buffer is not empty\n");

        abort();

    }



    f->buf[f->buf_index++] = v;

    f->is_write = 1;

    if (f->buf_index >= IO_BUF_SIZE) {

        int ret = qemu_fflush(f);

        qemu_file_set_if_error(f, ret);

    }

}
