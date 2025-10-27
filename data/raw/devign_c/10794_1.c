static void put_bool(QEMUFile *f, void *pv, size_t size)

{

    bool *v = pv;

    qemu_put_byte(f, *v);

}
