static void virtio_scsi_complete_req(VirtIOSCSIReq *req)

{

    VirtIOSCSI *s = req->dev;

    VirtQueue *vq = req->vq;

    VirtIODevice *vdev = VIRTIO_DEVICE(s);



    qemu_iovec_from_buf(&req->resp_iov, 0, &req->resp, req->resp_size);

    virtqueue_push(vq, &req->elem, req->qsgl.size + req->resp_iov.size);

    if (s->dataplane_started && !s->dataplane_fenced) {

        virtio_scsi_dataplane_notify(vdev, req);

    } else {

        virtio_notify(vdev, vq);

    }



    if (req->sreq) {

        req->sreq->hba_private = NULL;

        scsi_req_unref(req->sreq);

    }

    virtio_scsi_free_req(req);

}
