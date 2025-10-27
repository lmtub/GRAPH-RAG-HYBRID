int qemu_fclose(QEMUFile *f)

{

    int ret = 0;

    qemu_fflush(f);

    if (f->close)

        ret = f->close(f->opaque);

    g_free(f);

    return ret;

}
