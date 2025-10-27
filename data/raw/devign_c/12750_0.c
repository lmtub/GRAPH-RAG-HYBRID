static int get_int32_le(QEMUFile *f, void *pv, size_t size)

{

    int32_t *old = pv;

    int32_t new;

    qemu_get_sbe32s(f, &new);



    if (*old <= new) {

        return 0;

    }

    return -EINVAL;

}
