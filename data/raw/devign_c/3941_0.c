static void virtio_scsi_handle_event(VirtIODevice *vdev, VirtQueue *vq)

{

    VirtIOSCSI *s = VIRTIO_SCSI(vdev);



    if (s->ctx) {

        virtio_scsi_dataplane_start(s);

        if (!s->dataplane_fenced) {

            return;

        }

    }

    virtio_scsi_handle_event_vq(s, vq);

}
