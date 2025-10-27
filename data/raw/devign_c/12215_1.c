void qemu_put_be32(QEMUFile *f, unsigned int v)

{

    qemu_put_byte(f, v >> 24);

    qemu_put_byte(f, v >> 16);

    qemu_put_byte(f, v >> 8);

    qemu_put_byte(f, v);

}
