static int get_uint32_equal(QEMUFile *f, void *pv, size_t size)

{

    uint32_t *v = pv;

    uint32_t v2;

    qemu_get_be32s(f, &v2);



    if (*v == v2) {

        return 0;

    }

    return -EINVAL;

}
