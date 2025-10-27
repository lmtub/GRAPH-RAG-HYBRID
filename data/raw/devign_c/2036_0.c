static void glib_select_poll(fd_set *rfds, fd_set *wfds, fd_set *xfds,

                             bool err)

{

    GMainContext *context = g_main_context_default();



    if (!err) {

        int i;



        for (i = 0; i < n_poll_fds; i++) {

            GPollFD *p = &poll_fds[i];



            if ((p->events & G_IO_IN) && FD_ISSET(p->fd, rfds)) {

                p->revents |= G_IO_IN;

            }

            if ((p->events & G_IO_OUT) && FD_ISSET(p->fd, wfds)) {

                p->revents |= G_IO_OUT;

            }

            if ((p->events & G_IO_ERR) && FD_ISSET(p->fd, xfds)) {

                p->revents |= G_IO_ERR;

            }

        }

    }



    if (g_main_context_check(context, max_priority, poll_fds, n_poll_fds)) {

        g_main_context_dispatch(context);

    }

}
