static int s390_virtio_blk_init(VirtIOS390Device *s390_dev)

{

    VirtIOBlkS390 *dev = VIRTIO_BLK_S390(s390_dev);

    DeviceState *vdev = DEVICE(&dev->vdev);

    virtio_blk_set_conf(vdev, &(dev->blk));

    qdev_set_parent_bus(vdev, BUS(&s390_dev->bus));

    if (qdev_init(vdev) < 0) {

        return -1;

    }

    return s390_virtio_device_init(s390_dev, VIRTIO_DEVICE(vdev));

}
