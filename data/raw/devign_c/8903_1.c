static int s390_virtio_rng_init(VirtIOS390Device *s390_dev)

{

    VirtIORNGS390 *dev = VIRTIO_RNG_S390(s390_dev);

    DeviceState *vdev = DEVICE(&dev->vdev);



    qdev_set_parent_bus(vdev, BUS(&s390_dev->bus));

    if (qdev_init(vdev) < 0) {

        return -1;

    }



    object_property_set_link(OBJECT(dev),

                             OBJECT(dev->vdev.conf.default_backend), "rng",

                             NULL);



    return s390_virtio_device_init(s390_dev, VIRTIO_DEVICE(vdev));

}
