static void put_int16(QEMUFile *f, void *pv, size_t size)

{

    int16_t *v = pv;

    qemu_put_sbe16s(f, v);

}
