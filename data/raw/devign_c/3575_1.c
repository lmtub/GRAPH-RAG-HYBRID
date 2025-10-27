void virtio_scsi_dataplane_notify(VirtIODevice *vdev, VirtIOSCSIReq *req)

{

    if (virtio_should_notify(vdev, req->vq)) {

        event_notifier_set(virtio_queue_get_guest_notifier(req->vq));

    }

}
