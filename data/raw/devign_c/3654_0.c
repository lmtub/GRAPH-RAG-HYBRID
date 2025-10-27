int64_t qemu_fseek(QEMUFile *f, int64_t pos, int whence)

{

    if (whence == SEEK_SET) {

        /* nothing to do */

    } else if (whence == SEEK_CUR) {

        pos += qemu_ftell(f);

    } else {

        /* SEEK_END not supported */

        return -1;

    }

    if (f->is_writable) {

        qemu_fflush(f);

        f->buf_offset = pos;

    } else {

        f->buf_offset = pos;

        f->buf_index = 0;

        f->buf_size = 0;

    }

    return pos;

}
