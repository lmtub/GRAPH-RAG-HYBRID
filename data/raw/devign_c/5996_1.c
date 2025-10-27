static int vhost_scsi_exit(DeviceState *qdev)

{

    VirtIODevice *vdev = VIRTIO_DEVICE(qdev);

    VHostSCSI *s = VHOST_SCSI(qdev);

    VirtIOSCSICommon *vs = VIRTIO_SCSI_COMMON(qdev);



    migrate_del_blocker(s->migration_blocker);

    error_free(s->migration_blocker);



    /* This will stop vhost backend. */

    vhost_scsi_set_status(vdev, 0);



    g_free(s->dev.vqs);

    return virtio_scsi_common_exit(vs);

}
