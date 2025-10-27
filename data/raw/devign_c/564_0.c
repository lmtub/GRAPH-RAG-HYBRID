static int default_monitor_get_fd(Monitor *mon, const char *name, Error **errp)

{

    error_setg(errp, "only QEMU supports file descriptor passing");

    return -1;

}
