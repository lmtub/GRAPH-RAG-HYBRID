void qemu_fclose(QEMUFile *f)

{

    if (f->is_writable)

        qemu_fflush(f);

    if (f->is_file) {

        fclose(f->outfile);

    }

    qemu_free(f);

}
