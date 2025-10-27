static void gpollfds_from_select(void)

{

    int fd;

    for (fd = 0; fd <= nfds; fd++) {

        int events = 0;

        if (FD_ISSET(fd, &rfds)) {

            events |= G_IO_IN | G_IO_HUP | G_IO_ERR;

        }

        if (FD_ISSET(fd, &wfds)) {

            events |= G_IO_OUT | G_IO_ERR;

        }

        if (FD_ISSET(fd, &xfds)) {

            events |= G_IO_PRI;

        }

        if (events) {

            GPollFD pfd = {

                .fd = fd,

                .events = events,

            };

            g_array_append_val(gpollfds, pfd);

        }

    }

}
