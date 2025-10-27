static int virtio_blk_pci_init(VirtIOPCIProxy *vpci_dev)

{

    VirtIOBlkPCI *dev = VIRTIO_BLK_PCI(vpci_dev);

    DeviceState *vdev = DEVICE(&dev->vdev);

    virtio_blk_set_conf(vdev, &(dev->blk));

    qdev_set_parent_bus(vdev, BUS(&vpci_dev->bus));

    if (qdev_init(vdev) < 0) {

        return -1;

    }

    return 0;

}
