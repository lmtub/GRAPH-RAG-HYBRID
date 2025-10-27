static int get_uint16(QEMUFile *f, void *pv, size_t size)

{

    uint16_t *v = pv;

    qemu_get_be16s(f, v);

    return 0;

}
