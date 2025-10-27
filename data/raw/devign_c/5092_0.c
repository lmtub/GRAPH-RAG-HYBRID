static void virtio_scsi_hotplug(HotplugHandler *hotplug_dev, DeviceState *dev,

                                Error **errp)

{

    VirtIODevice *vdev = VIRTIO_DEVICE(hotplug_dev);

    VirtIOSCSI *s = VIRTIO_SCSI(vdev);

    SCSIDevice *sd = SCSI_DEVICE(dev);



    if (s->ctx && !s->dataplane_disabled) {

        if (blk_op_is_blocked(sd->conf.blk, BLOCK_OP_TYPE_DATAPLANE, errp)) {

            return;

        }

        blk_op_block_all(sd->conf.blk, s->blocker);

    }



    if ((vdev->guest_features >> VIRTIO_SCSI_F_HOTPLUG) & 1) {

        virtio_scsi_push_event(s, sd,

                               VIRTIO_SCSI_T_TRANSPORT_RESET,

                               VIRTIO_SCSI_EVT_RESET_RESCAN);

    }

}
