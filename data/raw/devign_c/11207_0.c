static int udp_read_packet(AVFormatContext *s, RTSPStream **prtsp_st,

                           uint8_t *buf, int buf_size, int64_t wait_end)

{

    RTSPState *rt = s->priv_data;

    RTSPStream *rtsp_st;

    int n, i, ret, timeout_cnt = 0;

    struct pollfd *p = rt->p;

    int *fds = NULL, fdsnum, fdsidx;



    if (!p) {

        p = rt->p = av_malloc_array(2 * (rt->nb_rtsp_streams + 1), sizeof(struct pollfd));

        if (!p)

            return AVERROR(ENOMEM);



        if (rt->rtsp_hd) {

            p[rt->max_p].fd = ffurl_get_file_handle(rt->rtsp_hd);

            p[rt->max_p++].events = POLLIN;

        }

        for (i = 0; i < rt->nb_rtsp_streams; i++) {

            rtsp_st = rt->rtsp_streams[i];

            if (rtsp_st->rtp_handle) {

                if (ret = ffurl_get_multi_file_handle(rtsp_st->rtp_handle,

                                                      &fds, &fdsnum)) {

                    av_log(s, AV_LOG_ERROR, "Unable to recover rtp ports\n");

                    return ret;

                }

                if (fdsnum != 2) {

                    av_log(s, AV_LOG_ERROR,

                           "Number of fds %d not supported\n", fdsnum);

                    return AVERROR_INVALIDDATA;

                }

                for (fdsidx = 0; fdsidx < fdsnum; fdsidx++) {

                    p[rt->max_p].fd       = fds[fdsidx];

                    p[rt->max_p++].events = POLLIN;

                }

                av_free(fds);

            }

        }

    }



    for (;;) {

        if (ff_check_interrupt(&s->interrupt_callback))

            return AVERROR_EXIT;

        if (wait_end && wait_end - av_gettime_relative() < 0)

            return AVERROR(EAGAIN);

        n = poll(p, rt->max_p, POLL_TIMEOUT_MS);

        if (n > 0) {

            int j = rt->rtsp_hd ? 1 : 0;

            timeout_cnt = 0;

            for (i = 0; i < rt->nb_rtsp_streams; i++) {

                rtsp_st = rt->rtsp_streams[i];

                if (rtsp_st->rtp_handle) {

                    if (p[j].revents & POLLIN || p[j+1].revents & POLLIN) {

                        ret = ffurl_read(rtsp_st->rtp_handle, buf, buf_size);

                        if (ret > 0) {

                            *prtsp_st = rtsp_st;

                            return ret;

                        }

                    }

                    j+=2;

                }

            }

#if CONFIG_RTSP_DEMUXER

            if (rt->rtsp_hd && p[0].revents & POLLIN) {

                return parse_rtsp_message(s);

            }

#endif

        } else if (n == 0 && ++timeout_cnt >= MAX_TIMEOUTS) {

            return AVERROR(ETIMEDOUT);

        } else if (n < 0 && errno != EINTR)

            return AVERROR(errno);

    }

}
