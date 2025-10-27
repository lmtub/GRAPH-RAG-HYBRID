static inline int retry_transfer_wrapper(URLContext *h, unsigned char *buf, int size, int size_min,

                                         int (*transfer_func)(URLContext *h, unsigned char *buf, int size))

{

    int ret, len;

    int fast_retries = 5;

    int64_t wait_since = 0;



    len = 0;

    while (len < size_min) {

        ret = transfer_func(h, buf+len, size-len);

        if (ret == AVERROR(EINTR))

            continue;

        if (h->flags & AVIO_FLAG_NONBLOCK)

            return ret;

        if (ret == AVERROR(EAGAIN)) {

            ret = 0;

            if (fast_retries) {

                fast_retries--;

            } else {

                if (h->rw_timeout) {

                    if (!wait_since)

                        wait_since = av_gettime();

                    else if (av_gettime() > wait_since + h->rw_timeout)

                        return AVERROR(EIO);

                }

                av_usleep(1000);

            }

        } else if (ret < 1)

            return (ret < 0 && ret != AVERROR_EOF) ? ret : len;

        if (ret)

           fast_retries = FFMAX(fast_retries, 2);

        len += ret;

        if (len < size && ff_check_interrupt(&h->interrupt_callback))

            return AVERROR_EXIT;

    }

    return len;

}
