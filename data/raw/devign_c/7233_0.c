static int pci_device_hot_remove(Monitor *mon, const char *pci_addr)

{

    PCIDevice *d;

    int dom, bus;

    unsigned slot;

    Error *local_err = NULL;



    if (pci_read_devaddr(mon, pci_addr, &dom, &bus, &slot)) {

        return -1;

    }



    d = pci_find_device(pci_find_root_bus(dom), bus, PCI_DEVFN(slot, 0));

    if (!d) {

        monitor_printf(mon, "slot %d empty\n", slot);

        return -1;

    }



    qdev_unplug(&d->qdev, &local_err);

    if (error_is_set(&local_err)) {

        monitor_printf(mon, "%s\n", error_get_pretty(local_err));

        error_free(local_err);

        return -1;

    }



    return 0;

}
