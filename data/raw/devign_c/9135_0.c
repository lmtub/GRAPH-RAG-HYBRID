static int udp_read_packet(AVFormatContext *s,

                           AVPacket *pkt)

{

    AVFormatContext *ic;

    AVStream *st;

    RTSPStream *rtsp_st;

    fd_set rfds;

    int fd1, fd2, fd_max, n, i, ret;

    char buf[RTP_MAX_PACKET_LENGTH];

    struct timeval tv;



    for(;;) {

        if (rtsp_abort_req)

            return -EIO;

        FD_ZERO(&rfds);

        fd_max = -1;

        for(i = 0; i < s->nb_streams; i++) {

            st = s->streams[i];

            rtsp_st = st->priv_data;

            ic = rtsp_st->ic;

            /* currently, we cannot probe RTCP handle because of blocking restrictions */

            rtp_get_file_handles(url_fileno(&ic->pb), &fd1, &fd2);

            if (fd1 > fd_max)

                fd_max = fd1;

            FD_SET(fd1, &rfds);

        }

        /* XXX: also add proper API to abort */

        tv.tv_sec = 0;

        tv.tv_usec = 500000;

        n = select(fd_max + 1, &rfds, NULL, NULL, &tv);

        if (n > 0) {

            for(i = 0; i < s->nb_streams; i++) {

                st = s->streams[i];

                rtsp_st = st->priv_data;

                ic = rtsp_st->ic;

                rtp_get_file_handles(url_fileno(&ic->pb), &fd1, &fd2);

                if (FD_ISSET(fd1, &rfds)) {

                    ret = url_read(url_fileno(&ic->pb), buf, sizeof(buf));

                    if (ret >= 0 && 

                        rtp_parse_packet(ic, pkt, buf, ret) == 0) {

                        pkt->stream_index = i;

                        return ret;

                    }

                }

            }

        }

    }

}
