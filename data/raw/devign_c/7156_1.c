static PCIDevice *qemu_pci_hot_add_nic(Monitor *mon,

                                       const char *devaddr,

                                       const char *opts_str)

{

    Error *local_err = NULL;

    QemuOpts *opts;

    PCIBus *root = pci_find_primary_bus();

    PCIBus *bus;

    int ret, devfn;



    if (!root) {

        monitor_printf(mon, "no primary PCI bus (if there are multiple"

                       " PCI roots, you must use device_add instead)");

        return NULL;

    }



    bus = pci_get_bus_devfn(&devfn, root, devaddr);

    if (!bus) {

        monitor_printf(mon, "Invalid PCI device address %s\n", devaddr);

        return NULL;

    }

    if (!qbus_is_hotpluggable(BUS(bus))) {

        monitor_printf(mon, "PCI bus doesn't support hotplug\n");

        return NULL;

    }



    opts = qemu_opts_parse(qemu_find_opts("net"), opts_str ? opts_str : "", 0);

    if (!opts) {

        return NULL;

    }



    qemu_opt_set(opts, "type", "nic");



    ret = net_client_init(opts, 0, &local_err);

    if (local_err) {

        qerror_report_err(local_err);

        error_free(local_err);

        return NULL;

    }

    if (nd_table[ret].devaddr) {

        monitor_printf(mon, "Parameter addr not supported\n");

        return NULL;

    }

    return pci_nic_init(&nd_table[ret], root, "rtl8139", devaddr);

}
