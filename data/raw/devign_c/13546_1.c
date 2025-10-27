int qemu_file_get_error(QEMUFile *f)

{

    return f->last_error;

}
