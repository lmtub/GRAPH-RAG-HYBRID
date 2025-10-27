int qemu_get_byte(QEMUFile *f)

{

    int result;



    result = qemu_peek_byte(f, 0);

    qemu_file_skip(f, 1);

    return result;

}
