bool virtio_scsi_handle_ctrl_vq(VirtIOSCSI *s, VirtQueue *vq)

{

    VirtIOSCSIReq *req;

    bool progress = false;



    virtio_scsi_acquire(s);

    while ((req = virtio_scsi_pop_req(s, vq))) {

        progress = true;

        virtio_scsi_handle_ctrl_req(s, req);

    }

    virtio_scsi_release(s);

    return progress;

}
