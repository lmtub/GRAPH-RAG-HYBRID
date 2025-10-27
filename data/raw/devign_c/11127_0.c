static void qemu_fill_buffer(QEMUFile *f)

{

    int len;



    if (f->is_writable)

        return;

    if (f->is_file) {

        fseek(f->outfile, f->buf_offset, SEEK_SET);

        len = fread(f->buf, 1, IO_BUF_SIZE, f->outfile);

        if (len < 0)

            len = 0;

    } else {

        len = bdrv_pread(f->bs, f->base_offset + f->buf_offset,

                         f->buf, IO_BUF_SIZE);

        if (len < 0)

            len = 0;

    }

    f->buf_index = 0;

    f->buf_size = len;

    f->buf_offset += len;

}
