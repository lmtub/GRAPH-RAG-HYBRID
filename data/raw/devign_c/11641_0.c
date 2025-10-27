static void virtio_net_pci_realize(VirtIOPCIProxy *vpci_dev, Error **errp)

{

    DeviceState *qdev = DEVICE(vpci_dev);

    VirtIONetPCI *dev = VIRTIO_NET_PCI(vpci_dev);

    DeviceState *vdev = DEVICE(&dev->vdev);



    virtio_net_set_config_size(&dev->vdev, vpci_dev->host_features);

    virtio_net_set_netclient_name(&dev->vdev, qdev->id,

                                  object_get_typename(OBJECT(qdev)));

    qdev_set_parent_bus(vdev, BUS(&vpci_dev->bus));

    object_property_set_bool(OBJECT(vdev), true, "realized", errp);

}
