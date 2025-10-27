static void pci_device_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *k = DEVICE_CLASS(klass);

    k->init = pci_qdev_init;

    k->exit = pci_unregister_device;

    k->bus_type = TYPE_PCI_BUS;

    k->props = pci_props;

}
