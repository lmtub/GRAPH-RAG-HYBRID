static void virtio_balloon_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    PCIDeviceClass *k = PCI_DEVICE_CLASS(klass);



    k->init = virtio_balloon_init_pci;

    k->exit = virtio_balloon_exit_pci;

    k->vendor_id = PCI_VENDOR_ID_REDHAT_QUMRANET;

    k->device_id = PCI_DEVICE_ID_VIRTIO_BALLOON;

    k->revision = VIRTIO_PCI_ABI_VERSION;

    k->class_id = PCI_CLASS_MEMORY_RAM;

    dc->alias = "virtio-balloon";

    dc->reset = virtio_pci_reset;

    dc->props = virtio_balloon_properties;

}
