static void virtio_scsi_device_realize(DeviceState *dev, Error **errp)

{

    VirtIODevice *vdev = VIRTIO_DEVICE(dev);

    VirtIOSCSI *s = VIRTIO_SCSI(dev);

    static int virtio_scsi_id;

    Error *err = NULL;



    virtio_scsi_common_realize(dev, &err);

    if (err != NULL) {

        error_propagate(errp, err);

        return;

    }



    scsi_bus_new(&s->bus, sizeof(s->bus), dev,

                 &virtio_scsi_scsi_info, vdev->bus_name);



    if (!dev->hotplugged) {

        scsi_bus_legacy_handle_cmdline(&s->bus, &err);

        if (err != NULL) {

            error_propagate(errp, err);

            return;

        }

    }



    register_savevm(dev, "virtio-scsi", virtio_scsi_id++, 1,

                    virtio_scsi_save, virtio_scsi_load, s);

}
