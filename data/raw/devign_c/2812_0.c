static int pci_qdev_init(DeviceState *qdev, DeviceInfo *base)

{

    PCIDevice *pci_dev = (PCIDevice *)qdev;

    PCIDeviceInfo *info = container_of(base, PCIDeviceInfo, qdev);

    PCIBus *bus;

    int devfn, rc;



    /* initialize cap_present for pci_is_express() and pci_config_size() */

    if (info->is_express) {

        pci_dev->cap_present |= QEMU_PCI_CAP_EXPRESS;

    }



    bus = FROM_QBUS(PCIBus, qdev_get_parent_bus(qdev));

    devfn = pci_dev->devfn;

    pci_dev = do_pci_register_device(pci_dev, bus, base->name, devfn,

                                     info->config_read, info->config_write);

    assert(pci_dev);

    rc = info->init(pci_dev);

    if (rc != 0)

        return rc;

    if (qdev->hotplugged)

        bus->hotplug(pci_dev, 1);

    return 0;

}
