static void virtio_serial_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    PCIDeviceClass *k = PCI_DEVICE_CLASS(klass);



    k->init = virtio_serial_init_pci;

    k->exit = virtio_serial_exit_pci;

    k->vendor_id = PCI_VENDOR_ID_REDHAT_QUMRANET;

    k->device_id = PCI_DEVICE_ID_VIRTIO_CONSOLE;

    k->revision = VIRTIO_PCI_ABI_VERSION;

    k->class_id = PCI_CLASS_COMMUNICATION_OTHER;

    dc->alias = "virtio-serial";

    dc->reset = virtio_pci_reset;

    dc->props = virtio_serial_properties;

}
