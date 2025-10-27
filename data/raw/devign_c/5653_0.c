static void virtio_net_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    PCIDeviceClass *k = PCI_DEVICE_CLASS(klass);



    k->init = virtio_net_init_pci;

    k->exit = virtio_net_exit_pci;

    k->romfile = "pxe-virtio.rom";

    k->vendor_id = PCI_VENDOR_ID_REDHAT_QUMRANET;

    k->device_id = PCI_DEVICE_ID_VIRTIO_NET;

    k->revision = VIRTIO_PCI_ABI_VERSION;

    k->class_id = PCI_CLASS_NETWORK_ETHERNET;

    dc->alias = "virtio-net";

    dc->reset = virtio_pci_reset;

    dc->props = virtio_net_properties;

}
