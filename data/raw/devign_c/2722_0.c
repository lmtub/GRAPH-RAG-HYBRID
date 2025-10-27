void qemu_iovec_concat(QEMUIOVector *dst,

                       QEMUIOVector *src, size_t soffset, size_t sbytes)

{

    int i;

    size_t done;

    struct iovec *siov = src->iov;

    assert(dst->nalloc != -1);

    assert(src->size >= soffset);

    for (i = 0, done = 0; done < sbytes && i < src->niov; i++) {

        if (soffset < siov[i].iov_len) {

            size_t len = MIN(siov[i].iov_len - soffset, sbytes - done);

            qemu_iovec_add(dst, siov[i].iov_base + soffset, len);

            done += len;

            soffset = 0;

        } else {

            soffset -= siov[i].iov_len;

        }

    }

    /* return done; */

}
