PCIDevice *pci_nic_init(NICInfo *nd, const char *default_model,

                        const char *default_devaddr)

{

    const char *devaddr = nd->devaddr ? nd->devaddr : default_devaddr;

    PCIBus *bus;

    int devfn;

    PCIDevice *pci_dev;

    DeviceState *dev;

    int i;



    i = qemu_find_nic_model(nd, pci_nic_models, default_model);

    if (i < 0)

        return NULL;



    bus = pci_get_bus_devfn(&devfn, devaddr);

    if (!bus) {

        error_report("Invalid PCI device address %s for device %s",

                     devaddr, pci_nic_names[i]);

        return NULL;

    }



    pci_dev = pci_create(bus, devfn, pci_nic_names[i]);

    dev = &pci_dev->qdev;

    if (nd->name)

        dev->id = qemu_strdup(nd->name);

    qdev_set_nic_properties(dev, nd);

    if (qdev_init(dev) < 0)

        return NULL;

    return pci_dev;

}
