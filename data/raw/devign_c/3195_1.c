static int get_uint32(QEMUFile *f, void *pv, size_t size)

{

    uint32_t *v = pv;

    qemu_get_be32s(f, v);

    return 0;

}
