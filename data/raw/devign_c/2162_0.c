static void virtio_ccw_net_realize(VirtioCcwDevice *ccw_dev, Error **errp)

{

    DeviceState *qdev = DEVICE(ccw_dev);

    VirtIONetCcw *dev = VIRTIO_NET_CCW(ccw_dev);

    DeviceState *vdev = DEVICE(&dev->vdev);

    Error *err = NULL;



    virtio_net_set_netclient_name(&dev->vdev, qdev->id,

                                  object_get_typename(OBJECT(qdev)));

    qdev_set_parent_bus(vdev, BUS(&ccw_dev->bus));

    object_property_set_bool(OBJECT(vdev), true, "realized", &err);

    if (err) {

        error_propagate(errp, err);

    }

}
