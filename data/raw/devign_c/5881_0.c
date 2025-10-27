static void virtio_ccw_scsi_realize(VirtioCcwDevice *ccw_dev, Error **errp)

{

    VirtIOSCSICcw *dev = VIRTIO_SCSI_CCW(ccw_dev);

    DeviceState *vdev = DEVICE(&dev->vdev);

    DeviceState *qdev = DEVICE(ccw_dev);

    Error *err = NULL;

    char *bus_name;



    /*

     * For command line compatibility, this sets the virtio-scsi-device bus

     * name as before.

     */

    if (qdev->id) {

        bus_name = g_strdup_printf("%s.0", qdev->id);

        virtio_device_set_child_bus_name(VIRTIO_DEVICE(vdev), bus_name);

        g_free(bus_name);

    }



    qdev_set_parent_bus(vdev, BUS(&ccw_dev->bus));

    object_property_set_bool(OBJECT(vdev), true, "realized", &err);

    if (err) {

        error_propagate(errp, err);

    }

}
