static void vhost_scsi_unrealize(DeviceState *dev, Error **errp)

{

    VirtIODevice *vdev = VIRTIO_DEVICE(dev);

    VHostSCSI *s = VHOST_SCSI(dev);



    migrate_del_blocker(s->migration_blocker);

    error_free(s->migration_blocker);



    /* This will stop vhost backend. */

    vhost_scsi_set_status(vdev, 0);




    g_free(s->dev.vqs);



    virtio_scsi_common_unrealize(dev, errp);

}