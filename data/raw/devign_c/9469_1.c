uint64_t qemu_get_be64(QEMUFile *f)

{

    uint64_t v;

    v = (uint64_t)qemu_get_be32(f) << 32;

    v |= qemu_get_be32(f);

    return v;

}
