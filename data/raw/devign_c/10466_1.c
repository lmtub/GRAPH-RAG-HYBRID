unsigned int qemu_get_be16(QEMUFile *f)

{

    unsigned int v;

    v = qemu_get_byte(f) << 8;

    v |= qemu_get_byte(f);

    return v;

}
