static int os_host_main_loop_wait(int64_t timeout)
{
    GMainContext *context = g_main_context_default();
    GPollFD poll_fds[1024 * 2]; /* this is probably overkill */
    int select_ret = 0;
    int g_poll_ret, ret, i, n_poll_fds;
    PollingEntry *pe;
    WaitObjects *w = &wait_objects;
    gint poll_timeout;
    int64_t poll_timeout_ns;
    static struct timeval tv0;
    fd_set rfds, wfds, xfds;
    int nfds;
    /* XXX: need to suppress polling by better using win32 events */
    ret = 0;
    for (pe = first_polling_entry; pe != NULL; pe = pe->next) {
        ret |= pe->func(pe->opaque);
    }
    if (ret != 0) {
        return ret;
    }
    FD_ZERO(&rfds);
    FD_ZERO(&wfds);
    FD_ZERO(&xfds);
    nfds = pollfds_fill(gpollfds, &rfds, &wfds, &xfds);
    if (nfds >= 0) {
        select_ret = select(nfds + 1, &rfds, &wfds, &xfds, &tv0);
        if (select_ret != 0) {
            timeout = 0;
        }
        if (select_ret > 0) {
            pollfds_poll(gpollfds, nfds, &rfds, &wfds, &xfds);
        }
    }
    g_main_context_prepare(context, &max_priority);
    n_poll_fds = g_main_context_query(context, max_priority, &poll_timeout,
                                      poll_fds, ARRAY_SIZE(poll_fds));
    g_assert(n_poll_fds <= ARRAY_SIZE(poll_fds));
    for (i = 0; i < w->num; i++) {
        poll_fds[n_poll_fds + i].fd = (DWORD_PTR)w->events[i];
        poll_fds[n_poll_fds + i].events = G_IO_IN;
    }
    if (poll_timeout < 0) {
        poll_timeout_ns = -1;
    } else {
        poll_timeout_ns = (int64_t)poll_timeout * (int64_t)SCALE_MS;
    }
    poll_timeout_ns = qemu_soonest_timeout(poll_timeout_ns, timeout);
    qemu_mutex_unlock_iothread();
    g_poll_ret = qemu_poll_ns(poll_fds, n_poll_fds + w->num, poll_timeout_ns);
    qemu_mutex_lock_iothread();
    if (g_poll_ret > 0) {
        for (i = 0; i < w->num; i++) {
            w->revents[i] = poll_fds[n_poll_fds + i].revents;
        }
        for (i = 0; i < w->num; i++) {
            if (w->revents[i] && w->func[i]) {
                w->func[i](w->opaque[i]);
            }
        }
    }
    if (g_main_context_check(context, max_priority, poll_fds, n_poll_fds)) {
        g_main_context_dispatch(context);
    }
    return select_ret || g_poll_ret;
}