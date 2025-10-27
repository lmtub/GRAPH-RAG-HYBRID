int virtio_bus_device_plugged(VirtIODevice *vdev)

{

    DeviceState *qdev = DEVICE(vdev);

    BusState *qbus = BUS(qdev_get_parent_bus(qdev));

    VirtioBusState *bus = VIRTIO_BUS(qbus);

    VirtioBusClass *klass = VIRTIO_BUS_GET_CLASS(bus);

    VirtioDeviceClass *vdc = VIRTIO_DEVICE_GET_CLASS(vdev);



    DPRINTF("%s: plug device.\n", qbus->name);



    if (klass->device_plugged != NULL) {

        klass->device_plugged(qbus->parent);

    }



    /* Get the features of the plugged device. */

    assert(vdc->get_features != NULL);

    vdev->host_features = vdc->get_features(vdev, vdev->host_features);

    return 0;

}
