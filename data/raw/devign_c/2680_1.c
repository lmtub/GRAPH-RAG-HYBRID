static int get_uint16_equal(QEMUFile *f, void *pv, size_t size)

{

    uint16_t *v = pv;

    uint16_t v2;

    qemu_get_be16s(f, &v2);



    if (*v == v2) {

        return 0;

    }

    return -EINVAL;

}
