static int get_unused_buffer(QEMUFile *f, void *pv, size_t size)

{

    qemu_fseek(f, size, SEEK_CUR);

    return 0;

}
