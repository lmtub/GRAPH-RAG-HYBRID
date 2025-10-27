static int pollfds_fill(GArray *pollfds, fd_set *rfds, fd_set *wfds,

                        fd_set *xfds)

{

    int nfds = -1;

    int i;



    for (i = 0; i < pollfds->len; i++) {

        GPollFD *pfd = &g_array_index(pollfds, GPollFD, i);

        int fd = pfd->fd;

        int events = pfd->events;

        if (events & G_IO_IN) {

            FD_SET(fd, rfds);

            nfds = MAX(nfds, fd);

        }

        if (events & G_IO_OUT) {

            FD_SET(fd, wfds);

            nfds = MAX(nfds, fd);

        }

        if (events & G_IO_PRI) {

            FD_SET(fd, xfds);

            nfds = MAX(nfds, fd);

        }

    }

    return nfds;

}
