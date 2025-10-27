void main_loop_wait(int nonblocking)

{

    fd_set rfds, wfds, xfds;

    int ret, nfds;

    struct timeval tv;

    int timeout;



    if (nonblocking)

        timeout = 0;

    else {

        timeout = qemu_calculate_timeout();

        qemu_bh_update_timeout(&timeout);

    }



    os_host_main_loop_wait(&timeout);



    tv.tv_sec = timeout / 1000;

    tv.tv_usec = (timeout % 1000) * 1000;



    /* poll any events */

    /* XXX: separate device handlers from system ones */

    nfds = -1;

    FD_ZERO(&rfds);

    FD_ZERO(&wfds);

    FD_ZERO(&xfds);

    qemu_iohandler_fill(&nfds, &rfds, &wfds, &xfds);

    slirp_select_fill(&nfds, &rfds, &wfds, &xfds);



    qemu_mutex_unlock_iothread();

    ret = select(nfds + 1, &rfds, &wfds, &xfds, &tv);

    qemu_mutex_lock_iothread();



    qemu_iohandler_poll(&rfds, &wfds, &xfds, ret);

    slirp_select_poll(&rfds, &wfds, &xfds, (ret < 0));



    qemu_run_all_timers();



    /* Check bottom-halves last in case any of the earlier events triggered

       them.  */

    qemu_bh_poll();



}
