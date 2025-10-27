static int get_uint64_as_uint32(QEMUFile *f, void *pv, size_t size,

                                VMStateField *field)

{

    uint64_t *v = pv;

    *v = qemu_get_be32(f);

    return 0;

}
