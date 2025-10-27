static int rtp_read(URLContext *h, uint8_t *buf, int size)

{

    RTPContext *s = h->priv_data;

    struct sockaddr_storage from;

    socklen_t from_len;

    int len, fd_max, n;

    fd_set rfds;

    struct timeval tv;

#if 0

    for(;;) {

        from_len = sizeof(from);

        len = recvfrom (s->rtp_fd, buf, size, 0,

                        (struct sockaddr *)&from, &from_len);

        if (len < 0) {

            if (ff_neterrno() == FF_NETERROR(EAGAIN) ||

                ff_neterrno() == FF_NETERROR(EINTR))

                continue;

            return AVERROR(EIO);

        }

        break;

    }

#else

    for(;;) {

        if (url_interrupt_cb())

            return AVERROR(EINTR);

        /* build fdset to listen to RTP and RTCP packets */

        FD_ZERO(&rfds);

        fd_max = s->rtp_fd;

        FD_SET(s->rtp_fd, &rfds);

        if (s->rtcp_fd > fd_max)

            fd_max = s->rtcp_fd;

        FD_SET(s->rtcp_fd, &rfds);

        tv.tv_sec = 0;

        tv.tv_usec = 100 * 1000;

        n = select(fd_max + 1, &rfds, NULL, NULL, &tv);

        if (n > 0) {

            /* first try RTCP */

            if (FD_ISSET(s->rtcp_fd, &rfds)) {

                from_len = sizeof(from);

                len = recvfrom (s->rtcp_fd, buf, size, 0,

                                (struct sockaddr *)&from, &from_len);

                if (len < 0) {

                    if (ff_neterrno() == FF_NETERROR(EAGAIN) ||

                        ff_neterrno() == FF_NETERROR(EINTR))

                        continue;

                    return AVERROR(EIO);

                }

                break;

            }

            /* then RTP */

            if (FD_ISSET(s->rtp_fd, &rfds)) {

                from_len = sizeof(from);

                len = recvfrom (s->rtp_fd, buf, size, 0,

                                (struct sockaddr *)&from, &from_len);

                if (len < 0) {

                    if (ff_neterrno() == FF_NETERROR(EAGAIN) ||

                        ff_neterrno() == FF_NETERROR(EINTR))

                        continue;

                    return AVERROR(EIO);

                }

                break;

            }

        } else if (n < 0) {

            if (ff_neterrno() == FF_NETERROR(EINTR))

                continue;

            return AVERROR(EIO);

        }

    }

#endif

    return len;

}
