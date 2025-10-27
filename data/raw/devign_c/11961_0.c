static int put_uint64_as_uint32(QEMUFile *f, void *pv, size_t size,

                                VMStateField *field, QJSON *vmdesc)

{

    uint64_t *v = pv;

    qemu_put_be32(f, *v);



    return 0;

}
