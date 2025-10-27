int qemu_file_rate_limit(QEMUFile *f)

{

    if (qemu_file_get_error(f)) {

        return 1;

    }

    if (f->xfer_limit > 0 && f->bytes_xfer > f->xfer_limit) {

        return 1;

    }

    return 0;

}
