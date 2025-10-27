void main_loop_wait(int nonblocking)

{

    IOHandlerRecord *ioh;

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



    /* poll any events */

    /* XXX: separate device handlers from system ones */

    nfds = -1;

    FD_ZERO(&rfds);

    FD_ZERO(&wfds);

    FD_ZERO(&xfds);

    QLIST_FOREACH(ioh, &io_handlers, next) {

        if (ioh->deleted)

            continue;

        if (ioh->fd_read &&

            (!ioh->fd_read_poll ||

             ioh->fd_read_poll(ioh->opaque) != 0)) {

            FD_SET(ioh->fd, &rfds);

            if (ioh->fd > nfds)

                nfds = ioh->fd;

        }

        if (ioh->fd_write) {

            FD_SET(ioh->fd, &wfds);

            if (ioh->fd > nfds)

                nfds = ioh->fd;

        }

    }



    tv.tv_sec = timeout / 1000;

    tv.tv_usec = (timeout % 1000) * 1000;



    slirp_select_fill(&nfds, &rfds, &wfds, &xfds);



    qemu_mutex_unlock_iothread();

    ret = select(nfds + 1, &rfds, &wfds, &xfds, &tv);

    qemu_mutex_lock_iothread();

    if (ret > 0) {

        IOHandlerRecord *pioh;



        QLIST_FOREACH_SAFE(ioh, &io_handlers, next, pioh) {

            if (ioh->deleted) {

                QLIST_REMOVE(ioh, next);

                qemu_free(ioh);

                continue;

            }

            if (ioh->fd_read && FD_ISSET(ioh->fd, &rfds)) {

                ioh->fd_read(ioh->opaque);

            }

            if (ioh->fd_write && FD_ISSET(ioh->fd, &wfds)) {

                ioh->fd_write(ioh->opaque);

            }

        }

    }



    slirp_select_poll(&rfds, &wfds, &xfds, (ret < 0));



    qemu_run_all_timers();



    /* Check bottom-halves last in case any of the earlier events triggered

       them.  */

    qemu_bh_poll();



}
