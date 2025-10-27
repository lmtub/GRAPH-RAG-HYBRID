static void add_to_iovec(QEMUFile *f, const uint8_t *buf, int size)

{

    /* check for adjacent buffer and coalesce them */

    if (f->iovcnt > 0 && buf == f->iov[f->iovcnt - 1].iov_base +

        f->iov[f->iovcnt - 1].iov_len) {

        f->iov[f->iovcnt - 1].iov_len += size;

    } else {

        f->iov[f->iovcnt].iov_base = (uint8_t *)buf;

        f->iov[f->iovcnt++].iov_len = size;

    }



    if (f->iovcnt >= MAX_IOV_SIZE) {

        qemu_fflush(f);

    }

}
