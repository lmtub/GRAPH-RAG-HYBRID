void virtio_scsi_handle_cmd_vq(VirtIOSCSI *s, VirtQueue *vq)

{

    VirtIOSCSIReq *req, *next;

    QTAILQ_HEAD(, VirtIOSCSIReq) reqs = QTAILQ_HEAD_INITIALIZER(reqs);



    while ((req = virtio_scsi_pop_req(s, vq))) {

        if (virtio_scsi_handle_cmd_req_prepare(s, req)) {

            QTAILQ_INSERT_TAIL(&reqs, req, next);

        }

    }



    QTAILQ_FOREACH_SAFE(req, &reqs, next, next) {

        virtio_scsi_handle_cmd_req_submit(s, req);

    }

}
