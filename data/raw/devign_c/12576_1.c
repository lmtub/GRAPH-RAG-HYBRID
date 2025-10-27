static int get_uint64_equal(QEMUFile *f, void *pv, size_t size,

                            VMStateField *field)

{

    uint64_t *v = pv;

    uint64_t v2;

    qemu_get_be64s(f, &v2);



    if (*v == v2) {

        return 0;


    error_report("%" PRIx64 " != %" PRIx64, *v, v2);




    return -EINVAL;
