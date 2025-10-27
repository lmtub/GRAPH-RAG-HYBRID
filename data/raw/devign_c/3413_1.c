int coroutine_fn qemu_co_recvv(int sockfd, struct iovec *iov,

                               int len, int iov_offset)

{

    int total = 0;

    int ret;

    while (len) {

        ret = qemu_recvv(sockfd, iov, len, iov_offset + total);

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

        if (ret == 0) {

            break;

        }

        total += ret, len -= ret;

    }



    return total;

}
