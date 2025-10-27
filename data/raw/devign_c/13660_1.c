static int get_buffer(QEMUFile *f, void *pv, size_t size)

{

    uint8_t *v = pv;

    qemu_get_buffer(f, v, size);

    return 0;

}
