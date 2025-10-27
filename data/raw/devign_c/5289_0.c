do_send_recv(int sockfd, struct iovec *iov, unsigned iov_cnt, bool do_send)

{

#if defined CONFIG_IOVEC && defined CONFIG_POSIX

    ssize_t ret;

    struct msghdr msg;

    memset(&msg, 0, sizeof(msg));

    msg.msg_iov = iov;

    msg.msg_iovlen = iov_cnt;

    do {

        ret = do_send

            ? sendmsg(sockfd, &msg, 0)

            : recvmsg(sockfd, &msg, 0);

    } while (ret < 0 && errno == EINTR);

    return ret;

#else

    /* else send piece-by-piece */

    /*XXX Note: windows has WSASend() and WSARecv() */

    unsigned i = 0;

    ssize_t ret = 0;

    while (i < iov_cnt) {

        ssize_t r = do_send

            ? send(sockfd, iov[i].iov_base, iov[i].iov_len, 0)

            : recv(sockfd, iov[i].iov_base, iov[i].iov_len, 0);

        if (r > 0) {

            ret += r;

        } else if (!r) {

            break;

        } else if (errno == EINTR) {

            continue;

        } else {

            /* else it is some "other" error,

             * only return if there was no data processed. */

            if (ret == 0) {

                ret = -1;

            }

            break;

        }

        i++;

    }

    return ret;

#endif

}
