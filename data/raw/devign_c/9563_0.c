PCIDevice *pci_try_create_multifunction(PCIBus *bus, int devfn,

                                        bool multifunction,

                                        const char *name)

{

    DeviceState *dev;



    dev = qdev_try_create(&bus->qbus, name);

    if (!dev) {

        return NULL;

    }

    qdev_prop_set_uint32(dev, "addr", devfn);

    qdev_prop_set_bit(dev, "multifunction", multifunction);

    return DO_UPCAST(PCIDevice, qdev, dev);

}
