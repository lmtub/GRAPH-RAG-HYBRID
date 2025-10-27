int ff_listen_connect(int fd, const struct sockaddr *addr,

                      socklen_t addrlen, int timeout, URLContext *h)

{

    struct pollfd p = {fd, POLLOUT, 0};

    int ret;

    socklen_t optlen;



    ff_socket_nonblock(fd, 1);



    while ((ret = connect(fd, addr, addrlen))) {

        ret = ff_neterrno();

        switch (ret) {

        case AVERROR(EINTR):

            if (ff_check_interrupt(&h->interrupt_callback))

                return AVERROR_EXIT;

            continue;

        case AVERROR(EINPROGRESS):

        case AVERROR(EAGAIN):

            while (timeout--) {

                if (ff_check_interrupt(&h->interrupt_callback))

                    return AVERROR_EXIT;

                ret = poll(&p, 1, 100);

                if (ret > 0)

                    break;

            }

            if (ret <= 0)

                return AVERROR(ETIMEDOUT);

            optlen = sizeof(ret);

            if (getsockopt (fd, SOL_SOCKET, SO_ERROR, &ret, &optlen))

                ret = AVUNERROR(ff_neterrno());

            if (ret != 0) {

                char errbuf[100];

                ret = AVERROR(ret);

                av_strerror(ret, errbuf, sizeof(errbuf));

                av_log(h, AV_LOG_ERROR,

                       "Connection to %s failed: %s\n",

                       h->filename, errbuf);

            }

        default:

            return ret;

        }

    }

    return ret;

}
