int event_notifier_init(EventNotifier *e, int active)

{

#ifdef CONFIG_EVENTFD

    int fd = eventfd(!!active, EFD_NONBLOCK | EFD_CLOEXEC);

    if (fd < 0)

        return -errno;

    e->fd = fd;

    return 0;

#else

    return -ENOSYS;

#endif

}
