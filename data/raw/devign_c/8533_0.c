int ff_network_wait_fd_timeout(int fd, int write, int64_t timeout, AVIOInterruptCB *int_cb)

{

    int ret;

    int64_t wait_start = 0;



    while (1) {

        ret = ff_network_wait_fd(fd, write);

        if (ret != AVERROR(EAGAIN))

            return ret;

        if (ff_check_interrupt(int_cb))

            return AVERROR_EXIT;

        if (timeout > 0) {

            if (!wait_start)

                wait_start = av_gettime();

            else if (av_gettime() - wait_start > timeout)

                return AVERROR(ETIMEDOUT);

        }

    }

}
