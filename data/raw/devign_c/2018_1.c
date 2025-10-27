static int net_vhost_check_net(void *opaque, QemuOpts *opts, Error **errp)

{

    const char *name = opaque;

    const char *driver, *netdev;

    const char virtio_name[] = "virtio-net-";



    driver = qemu_opt_get(opts, "driver");

    netdev = qemu_opt_get(opts, "netdev");



    if (!driver || !netdev) {

        return 0;

    }



    if (strcmp(netdev, name) == 0 &&

        strncmp(driver, virtio_name, strlen(virtio_name)) != 0) {

        error_report("vhost-user requires frontend driver virtio-net-*");

        return -1;

    }



    return 0;

}
