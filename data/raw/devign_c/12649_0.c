VirtIOSCSIReq *virtio_scsi_pop_req_vring(VirtIOSCSI *s,

                                         VirtIOSCSIVring *vring)

{

    VirtIOSCSIReq *req = virtio_scsi_init_req(s, NULL);

    int r;



    req->vring = vring;

    r = vring_pop((VirtIODevice *)s, &vring->vring, &req->elem);

    if (r < 0) {

        virtio_scsi_free_req(req);

        req = NULL;

    }

    return req;

}
