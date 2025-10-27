static void virtio_scsi_handle_cmd(VirtIODevice *vdev, VirtQueue *vq)

{

    /* use non-QOM casts in the data path */

    VirtIOSCSI *s = (VirtIOSCSI *)vdev;

    VirtIOSCSICommon *vs = &s->parent_obj;



    VirtIOSCSIReq *req;

    int n;



    while ((req = virtio_scsi_pop_req(s, vq))) {

        SCSIDevice *d;

        int out_size, in_size;

        if (req->elem.out_num < 1 || req->elem.in_num < 1) {

            virtio_scsi_bad_req();

        }



        out_size = req->elem.out_sg[0].iov_len;

        in_size = req->elem.in_sg[0].iov_len;

        if (out_size < sizeof(VirtIOSCSICmdReq) + vs->cdb_size ||

            in_size < sizeof(VirtIOSCSICmdResp) + vs->sense_size) {

            virtio_scsi_bad_req();

        }



        if (req->elem.out_num > 1 && req->elem.in_num > 1) {

            virtio_scsi_fail_cmd_req(req);

            continue;

        }



        d = virtio_scsi_device_find(s, req->req.cmd->lun);

        if (!d) {

            req->resp.cmd->response = VIRTIO_SCSI_S_BAD_TARGET;

            virtio_scsi_complete_req(req);

            continue;

        }

        req->sreq = scsi_req_new(d, req->req.cmd->tag,

                                 virtio_scsi_get_lun(req->req.cmd->lun),

                                 req->req.cmd->cdb, req);



        if (req->sreq->cmd.mode != SCSI_XFER_NONE) {

            int req_mode =

                (req->elem.in_num > 1 ? SCSI_XFER_FROM_DEV : SCSI_XFER_TO_DEV);



            if (req->sreq->cmd.mode != req_mode ||

                req->sreq->cmd.xfer > req->qsgl.size) {

                req->resp.cmd->response = VIRTIO_SCSI_S_OVERRUN;

                virtio_scsi_complete_req(req);

                continue;

            }

        }



        n = scsi_req_enqueue(req->sreq);

        if (n) {

            scsi_req_continue(req->sreq);

        }

    }

}
