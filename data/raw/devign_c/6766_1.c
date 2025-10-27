static void put_uint32(QEMUFile *f, void *pv, size_t size)

{

    uint32_t *v = pv;

    qemu_put_be32s(f, v);

}
