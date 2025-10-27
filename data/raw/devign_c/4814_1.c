int do_netdev_add(Monitor *mon, const QDict *qdict, QObject **ret_data)
{
    QemuOpts *opts;
    int res;
    opts = qemu_opts_from_qdict(&qemu_netdev_opts, qdict);
    if (!opts) {
        return -1;
    res = net_client_init(mon, opts, 1);
    return res;