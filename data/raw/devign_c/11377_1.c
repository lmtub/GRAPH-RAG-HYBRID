static int get_int16(QEMUFile *f, void *pv, size_t size)

{

    int16_t *v = pv;

    qemu_get_sbe16s(f, v);

    return 0;

}
