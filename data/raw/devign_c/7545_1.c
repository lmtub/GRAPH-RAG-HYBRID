ssize_t qsb_write_at(QEMUSizedBuffer *qsb, const uint8_t *source,

                     off_t pos, size_t count)

{

    ssize_t rc = qsb_grow(qsb, pos + count);

    size_t to_copy;

    size_t all_copy = count;

    const struct iovec *iov;

    ssize_t index;

    char *dest;

    off_t d_off, s_off = 0;



    if (rc < 0) {

        return rc;

    }



    if (pos + count > qsb->used) {

        qsb->used = pos + count;

    }



    index = qsb_get_iovec(qsb, pos, &d_off);

    if (index < 0) {

        return -EINVAL;

    }



    while (all_copy > 0) {

        iov = &qsb->iov[index];



        dest = iov->iov_base;



        to_copy = iov->iov_len - d_off;

        if (to_copy > all_copy) {

            to_copy = all_copy;

        }



        memcpy(&dest[d_off], &source[s_off], to_copy);



        s_off += to_copy;

        all_copy -= to_copy;



        d_off = 0;

        index++;

    }



    return count;

}
