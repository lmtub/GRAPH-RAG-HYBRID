int coroutine_fn qemu_co_sendv(int sockfd, struct iovec *iov,

                               int len, int iov_offset)

{

    int total = 0;

    int ret;

    while (len) {

        ret = qemu_sendv(sockfd, iov, len, iov_offset + total);

        if (ret < 0) {

            if (errno == EAGAIN) {

                qemu_coroutine_yield();

                continue;

            }

            if (total == 0) {

                total = -1;

            }

            break;

        }

        total += ret, len -= ret;

    }



    return total;

}
