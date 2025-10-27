static void virtio_blk_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    PCIDeviceClass *k = PCI_DEVICE_CLASS(klass);



    k->init = virtio_blk_init_pci;

    k->exit = virtio_blk_exit_pci;

    k->vendor_id = PCI_VENDOR_ID_REDHAT_QUMRANET;

    k->device_id = PCI_DEVICE_ID_VIRTIO_BLOCK;

    k->revision = VIRTIO_PCI_ABI_VERSION;

    k->class_id = PCI_CLASS_STORAGE_SCSI;

    dc->alias = "virtio-blk";

    dc->reset = virtio_pci_reset;

    dc->props = virtio_blk_properties;

}
