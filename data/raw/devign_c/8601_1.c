static struct iovec *adjust_sg(struct iovec *sg, int len, int *iovcnt)

{

    while (len && *iovcnt) {

        if (len < sg->iov_len) {

            sg->iov_len -= len;

            sg->iov_base += len;

            len = 0;

        } else {

            len -= sg->iov_len;

            sg++;

            *iovcnt -= 1;

        }

    }



    return sg;

}
