int64_t qemu_file_get_rate_limit(QEMUFile *f)

{

    return f->xfer_limit;

}
