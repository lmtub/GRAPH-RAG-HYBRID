static void virtio_scsi_change(SCSIBus *bus, SCSIDevice *dev, SCSISense sense)

{

    VirtIOSCSI *s = container_of(bus, VirtIOSCSI, bus);



    if (((s->vdev.guest_features >> VIRTIO_SCSI_F_CHANGE) & 1) &&

        (s->vdev.status & VIRTIO_CONFIG_S_DRIVER_OK) &&

        dev->type != TYPE_ROM) {

        virtio_scsi_push_event(s, dev, VIRTIO_SCSI_T_PARAM_CHANGE,

                               sense.asc | (sense.ascq << 8));

    }

}
