static int pci_device_hot_remove(Monitor *mon, const char *pci_addr)

{

    PCIBus *root = pci_find_primary_bus();

    PCIDevice *d;

    int bus;

    unsigned slot;

    Error *local_err = NULL;



    if (!root) {

        monitor_printf(mon, "no primary PCI bus (if there are multiple"

                       " PCI roots, you must use device_del instead)");

        return -1;

    }



    if (pci_read_devaddr(mon, pci_addr, &bus, &slot)) {

        return -1;

    }



    d = pci_find_device(root, bus, PCI_DEVFN(slot, 0));

    if (!d) {

        monitor_printf(mon, "slot %d empty\n", slot);

        return -1;

    }



    qdev_unplug(&d->qdev, &local_err);

    if (local_err) {

        monitor_printf(mon, "%s\n", error_get_pretty(local_err));

        error_free(local_err);

        return -1;

    }



    return 0;

}
