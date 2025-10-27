static void init_virtio_dev(TestServer *s)

{

    QPCIBus *bus;

    QVirtioPCIDevice *dev;

    uint32_t features;



    bus = qpci_init_pc(NULL);

    g_assert_nonnull(bus);



    dev = qvirtio_pci_device_find(bus, VIRTIO_ID_NET);

    g_assert_nonnull(dev);



    qvirtio_pci_device_enable(dev);

    qvirtio_reset(&dev->vdev);

    qvirtio_set_acknowledge(&dev->vdev);

    qvirtio_set_driver(&dev->vdev);



    features = qvirtio_get_features(&dev->vdev);

    features = features & VIRTIO_NET_F_MAC;

    qvirtio_set_features(&dev->vdev, features);



    qvirtio_set_driver_ok(&dev->vdev);

}
