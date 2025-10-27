int event_notifier_set(EventNotifier *e)

{

    uint64_t value = 1;

    int r = write(e->fd, &value, sizeof(value));

    return r == sizeof(value);

}
