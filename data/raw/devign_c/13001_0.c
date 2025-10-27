static void virtio_input_pci_realize(VirtIOPCIProxy *vpci_dev, Error **errp)

{

    VirtIOInputPCI *vinput = VIRTIO_INPUT_PCI(vpci_dev);

    DeviceState *vdev = DEVICE(&vinput->vdev);



    qdev_set_parent_bus(vdev, BUS(&vpci_dev->bus));

    /* force virtio-1.0 */

    vpci_dev->flags &= ~VIRTIO_PCI_FLAG_DISABLE_MODERN;

    vpci_dev->flags |= VIRTIO_PCI_FLAG_DISABLE_LEGACY;

    object_property_set_bool(OBJECT(vdev), true, "realized", errp);

}
