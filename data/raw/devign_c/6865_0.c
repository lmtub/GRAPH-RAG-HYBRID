QEMUFile *qemu_fopen(const char *filename, const char *mode)

{

    QEMUFile *f;



    f = qemu_mallocz(sizeof(QEMUFile));

    if (!f)

        return NULL;

    if (!strcmp(mode, "wb")) {

        f->is_writable = 1;

    } else if (!strcmp(mode, "rb")) {

        f->is_writable = 0;

    } else {

        goto fail;

    }

    f->outfile = fopen(filename, mode);

    if (!f->outfile)

        goto fail;

    f->is_file = 1;

    return f;

 fail:

    if (f->outfile)

        fclose(f->outfile);

    qemu_free(f);

    return NULL;

}
