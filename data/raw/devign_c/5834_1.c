static int get_int64(QEMUFile *f, void *pv, size_t size)

{

    int64_t *v = pv;

    qemu_get_sbe64s(f, v);

    return 0;

}
