static int get_uint8(QEMUFile *f, void *pv, size_t size)

{

    uint8_t *v = pv;

    qemu_get_8s(f, v);

    return 0;

}
