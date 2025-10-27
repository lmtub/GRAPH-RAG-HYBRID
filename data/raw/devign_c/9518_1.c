ssize_t iov_send_recv(int sockfd, struct iovec *iov,

                      size_t offset, size_t bytes,

                      bool do_sendv)

{

    int iovlen;

    ssize_t ret;

    size_t diff;

    struct iovec *last_iov;



    /* last_iov is inclusive, so count from one.  */

    iovlen = 1;

    last_iov = iov;

    bytes += offset;



    while (last_iov->iov_len < bytes) {

        bytes -= last_iov->iov_len;



        last_iov++;

        iovlen++;

    }



    diff = last_iov->iov_len - bytes;

    last_iov->iov_len -= diff;



    while (iov->iov_len <= offset) {

        offset -= iov->iov_len;



        iov++;

        iovlen--;

    }



    iov->iov_base = (char *) iov->iov_base + offset;

    iov->iov_len -= offset;



    {

#if defined CONFIG_IOVEC && defined CONFIG_POSIX

        struct msghdr msg;

        memset(&msg, 0, sizeof(msg));

        msg.msg_iov = iov;

        msg.msg_iovlen = iovlen;



        do {

            if (do_sendv) {

                ret = sendmsg(sockfd, &msg, 0);

            } else {

                ret = recvmsg(sockfd, &msg, 0);

            }

        } while (ret == -1 && errno == EINTR);

#else

        struct iovec *p = iov;

        ret = 0;

        while (iovlen > 0) {

            int rc;

            if (do_sendv) {

                rc = send(sockfd, p->iov_base, p->iov_len, 0);

            } else {

                rc = qemu_recv(sockfd, p->iov_base, p->iov_len, 0);

            }

            if (rc == -1) {

                if (errno == EINTR) {

                    continue;

                }

                if (ret == 0) {

                    ret = -1;

                }

                break;

            }

            if (rc == 0) {

                break;

            }

            ret += rc;

            iovlen--, p++;

        }

#endif

    }



    /* Undo the changes above */

    iov->iov_base = (char *) iov->iov_base - offset;

    iov->iov_len += offset;

    last_iov->iov_len += diff;

    return ret;

}
