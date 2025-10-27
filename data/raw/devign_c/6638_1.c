static bool xen_host_pci_dev_is_virtfn(XenHostPCIDevice *d)

{

    char path[PATH_MAX];

    struct stat buf;



    if (xen_host_pci_sysfs_path(d, "physfn", path, sizeof (path))) {

        return false;

    }

    return !stat(path, &buf);

}
