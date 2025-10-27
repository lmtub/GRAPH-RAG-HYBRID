ssize_t qsb_get_buffer(const QEMUSizedBuffer *qsb, off_t start,

                       size_t count, uint8_t *buffer)

{

    const struct iovec *iov;

    size_t to_copy, all_copy;

    ssize_t index;

    off_t s_off;

    off_t d_off = 0;

    char *s;



    if (start > qsb->used) {

        return 0;

    }



    all_copy = qsb->used - start;

    if (all_copy > count) {

        all_copy = count;

    } else {

        count = all_copy;

    }



    index = qsb_get_iovec(qsb, start, &s_off);

    if (index < 0) {

        return 0;

    }



    while (all_copy > 0) {

        iov = &qsb->iov[index];



        s = iov->iov_base;



        to_copy = iov->iov_len - s_off;

        if (to_copy > all_copy) {

            to_copy = all_copy;

        }

        memcpy(&buffer[d_off], &s[s_off], to_copy);



        d_off += to_copy;

        all_copy -= to_copy;



        s_off = 0;

        index++;

    }



    return count;

}
