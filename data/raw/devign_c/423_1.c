static int xen_host_pci_config_open(XenHostPCIDevice *d)

{

    char path[PATH_MAX];

    int rc;



    rc = xen_host_pci_sysfs_path(d, "config", path, sizeof (path));

    if (rc) {

        return rc;

    }

    d->config_fd = open(path, O_RDWR);

    if (d->config_fd < 0) {

        return -errno;

    }

    return 0;

}
