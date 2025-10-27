static int virtio_rng_pci_init(VirtIOPCIProxy *vpci_dev)

{

    VirtIORngPCI *vrng = VIRTIO_RNG_PCI(vpci_dev);

    DeviceState *vdev = DEVICE(&vrng->vdev);



    qdev_set_parent_bus(vdev, BUS(&vpci_dev->bus));

    if (qdev_init(vdev) < 0) {

        return -1;

    }



    object_property_set_link(OBJECT(vrng),

                             OBJECT(vrng->vdev.conf.default_backend), "rng",

                             NULL);



    return 0;

}
