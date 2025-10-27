static int get_uint64_equal(QEMUFile *f, void *pv, size_t size)

{

    uint64_t *v = pv;

    uint64_t v2;

    qemu_get_be64s(f, &v2);



    if (*v == v2) {

        return 0;

    }

    return -EINVAL;

}
