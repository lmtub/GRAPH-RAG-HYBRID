static void monitor_fdset_cleanup(MonFdset *mon_fdset)

{

    MonFdsetFd *mon_fdset_fd;

    MonFdsetFd *mon_fdset_fd_next;



    QLIST_FOREACH_SAFE(mon_fdset_fd, &mon_fdset->fds, next, mon_fdset_fd_next) {

        if (mon_fdset_fd->removed ||

                (QLIST_EMPTY(&mon_fdset->dup_fds) && mon_refcount == 0)) {

            close(mon_fdset_fd->fd);

            g_free(mon_fdset_fd->opaque);

            QLIST_REMOVE(mon_fdset_fd, next);

            g_free(mon_fdset_fd);

        }

    }



    if (QLIST_EMPTY(&mon_fdset->fds) && QLIST_EMPTY(&mon_fdset->dup_fds)) {

        QLIST_REMOVE(mon_fdset, next);

        g_free(mon_fdset);

    }

}
