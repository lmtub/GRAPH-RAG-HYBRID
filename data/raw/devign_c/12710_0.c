static void virtio_scsi_complete_req(VirtIOSCSIReq *req)

{

    VirtIOSCSI *s = req->dev;

    VirtQueue *vq = req->vq;

    VirtIODevice *vdev = VIRTIO_DEVICE(s);

    virtqueue_push(vq, &req->elem, req->qsgl.size + req->elem.in_sg[0].iov_len);

    if (req->sreq) {

        req->sreq->hba_private = NULL;

        scsi_req_unref(req->sreq);

    }

    virtio_scsi_free_req(req);

    virtio_notify(vdev, vq);

}
