static int get_uint64(QEMUFile *f, void *pv, size_t size)

{

    uint64_t *v = pv;

    qemu_get_be64s(f, v);

    return 0;

}
