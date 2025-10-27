void qemu_fflush(QEMUFile *f)

{

    if (!f->is_writable)

        return;

    if (f->buf_index > 0) {

        if (f->is_file) {

            fseek(f->outfile, f->buf_offset, SEEK_SET);

            fwrite(f->buf, 1, f->buf_index, f->outfile);

        } else {

            bdrv_pwrite(f->bs, f->base_offset + f->buf_offset,

                        f->buf, f->buf_index);

        }

        f->buf_offset += f->buf_index;

        f->buf_index = 0;

    }

}
