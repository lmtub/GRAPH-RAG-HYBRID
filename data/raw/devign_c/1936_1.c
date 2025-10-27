static int udp_read_packet(AVFormatContext *s, RTSPStream **prtsp_st,

                           uint8_t *buf, int buf_size)

{

    RTSPState *rt = s->priv_data;

    RTSPStream *rtsp_st;

    fd_set rfds;

    int fd, fd_max, n, i, ret, tcp_fd;

    struct timeval tv;



    for(;;) {

        if (url_interrupt_cb())

            return AVERROR(EINTR);

        FD_ZERO(&rfds);

        tcp_fd = fd_max = url_get_file_handle(rt->rtsp_hd);

        FD_SET(tcp_fd, &rfds);

        for(i = 0; i < rt->nb_rtsp_streams; i++) {

            rtsp_st = rt->rtsp_streams[i];

            if (rtsp_st->rtp_handle) {

                /* currently, we cannot probe RTCP handle because of

                 * blocking restrictions */

                fd = url_get_file_handle(rtsp_st->rtp_handle);

                if (fd > fd_max)

                    fd_max = fd;

                FD_SET(fd, &rfds);

            }

        }

        tv.tv_sec = 0;

        tv.tv_usec = 100 * 1000;

        n = select(fd_max + 1, &rfds, NULL, NULL, &tv);

        if (n > 0) {

            for(i = 0; i < rt->nb_rtsp_streams; i++) {

                rtsp_st = rt->rtsp_streams[i];

                if (rtsp_st->rtp_handle) {

                    fd = url_get_file_handle(rtsp_st->rtp_handle);

                    if (FD_ISSET(fd, &rfds)) {

                        ret = url_read(rtsp_st->rtp_handle, buf, buf_size);

                        if (ret > 0) {

                            *prtsp_st = rtsp_st;

                            return ret;

                        }

                    }

                }

            }

            if (FD_ISSET(tcp_fd, &rfds)) {

                RTSPMessageHeader reply;



                rtsp_read_reply(s, &reply, NULL, 0);

                /* XXX: parse message */

            }

        }

    }

}
