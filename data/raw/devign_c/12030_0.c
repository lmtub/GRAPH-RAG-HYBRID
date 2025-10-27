static void vhost_ccw_scsi_realize(VirtioCcwDevice *ccw_dev, Error **errp)

{

    VHostSCSICcw *dev = VHOST_SCSI_CCW(ccw_dev);

    DeviceState *vdev = DEVICE(&dev->vdev);

    Error *err = NULL;



    qdev_set_parent_bus(vdev, BUS(&ccw_dev->bus));

    object_property_set_bool(OBJECT(vdev), true, "realized", &err);

    if (err) {

        error_propagate(errp, err);

    }

}
