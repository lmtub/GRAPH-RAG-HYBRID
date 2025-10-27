static void virtio_scsi_handle_ctrl(VirtIODevice *vdev, VirtQueue *vq)

{

    VirtIOSCSI *s = (VirtIOSCSI *)vdev;

    VirtIOSCSIReq *req;



    if (s->ctx && !s->dataplane_started) {

        virtio_scsi_dataplane_start(s);

        return;

    }

    while ((req = virtio_scsi_pop_req(s, vq))) {

        virtio_scsi_handle_ctrl_req(s, req);

    }

}
