void main_loop_wait(int timeout)

{

    IOHandlerRecord *ioh;

    fd_set rfds, wfds, xfds;

    int ret, nfds;

#ifdef _WIN32

    int ret2, i;

#endif

    struct timeval tv;

    PollingEntry *pe;





    /* XXX: need to suppress polling by better using win32 events */

    ret = 0;

    for(pe = first_polling_entry; pe != NULL; pe = pe->next) {

        ret |= pe->func(pe->opaque);

    }

#ifdef _WIN32

    if (ret == 0) {

        int err;

        WaitObjects *w = &wait_objects;



        ret = WaitForMultipleObjects(w->num, w->events, FALSE, timeout);

        if (WAIT_OBJECT_0 + 0 <= ret && ret <= WAIT_OBJECT_0 + w->num - 1) {

            if (w->func[ret - WAIT_OBJECT_0])

                w->func[ret - WAIT_OBJECT_0](w->opaque[ret - WAIT_OBJECT_0]);



            /* Check for additional signaled events */

            for(i = (ret - WAIT_OBJECT_0 + 1); i < w->num; i++) {



                /* Check if event is signaled */

                ret2 = WaitForSingleObject(w->events[i], 0);

                if(ret2 == WAIT_OBJECT_0) {

                    if (w->func[i])

                        w->func[i](w->opaque[i]);

                } else if (ret2 == WAIT_TIMEOUT) {

                } else {

                    err = GetLastError();

                    fprintf(stderr, "WaitForSingleObject error %d %d\n", i, err);

                }

            }

        } else if (ret == WAIT_TIMEOUT) {

        } else {

            err = GetLastError();

            fprintf(stderr, "WaitForMultipleObjects error %d %d\n", ret, err);

        }

    }

#endif

    /* poll any events */

    /* XXX: separate device handlers from system ones */

    nfds = -1;

    FD_ZERO(&rfds);

    FD_ZERO(&wfds);

    FD_ZERO(&xfds);

    for(ioh = first_io_handler; ioh != NULL; ioh = ioh->next) {

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



    tv.tv_sec = 0;

#ifdef _WIN32

    tv.tv_usec = 0;

#else

    tv.tv_usec = timeout * 1000;

#endif

#if defined(CONFIG_SLIRP)

    if (slirp_inited) {

        slirp_select_fill(&nfds, &rfds, &wfds, &xfds);

    }

#endif

    ret = select(nfds + 1, &rfds, &wfds, &xfds, &tv);

    if (ret > 0) {

        IOHandlerRecord **pioh;



        for(ioh = first_io_handler; ioh != NULL; ioh = ioh->next) {

            if (!ioh->deleted && ioh->fd_read && FD_ISSET(ioh->fd, &rfds)) {

                ioh->fd_read(ioh->opaque);

            }

            if (!ioh->deleted && ioh->fd_write && FD_ISSET(ioh->fd, &wfds)) {

                ioh->fd_write(ioh->opaque);

            }

        }



	/* remove deleted IO handlers */

	pioh = &first_io_handler;

	while (*pioh) {

            ioh = *pioh;

            if (ioh->deleted) {

                *pioh = ioh->next;

                qemu_free(ioh);

            } else

                pioh = &ioh->next;

        }

    }

#if defined(CONFIG_SLIRP)

    if (slirp_inited) {

        if (ret < 0) {

            FD_ZERO(&rfds);

            FD_ZERO(&wfds);

            FD_ZERO(&xfds);

        }

        slirp_select_poll(&rfds, &wfds, &xfds);

    }

#endif

    qemu_aio_poll();



    if (vm_running) {

        if (likely(!(cur_cpu->singlestep_enabled & SSTEP_NOTIMER)))

        qemu_run_timers(&active_timers[QEMU_TIMER_VIRTUAL],

                        qemu_get_clock(vm_clock));

        /* run dma transfers, if any */

        DMA_run();

    }



    /* real time timers */

    qemu_run_timers(&active_timers[QEMU_TIMER_REALTIME],

                    qemu_get_clock(rt_clock));



    if (alarm_timer->flags & ALARM_FLAG_EXPIRED) {

        alarm_timer->flags &= ~(ALARM_FLAG_EXPIRED);

        qemu_rearm_alarm_timer(alarm_timer);

    }



    /* Check bottom-halves last in case any of the earlier events triggered

       them.  */

    qemu_bh_poll();



}
