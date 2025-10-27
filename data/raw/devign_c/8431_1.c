static void put_uint8(QEMUFile *f, void *pv, size_t size)

{

    uint8_t *v = pv;

    qemu_put_8s(f, v);

}
