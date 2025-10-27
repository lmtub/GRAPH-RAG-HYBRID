static void virtio_scsi_request_cancelled(SCSIRequest *r)

{

    VirtIOSCSIReq *req = r->hba_private;



    if (!req) {

        return;

    }

    if (req->dev->resetting) {

        req->resp.cmd->response = VIRTIO_SCSI_S_RESET;

    } else {

        req->resp.cmd->response = VIRTIO_SCSI_S_ABORTED;

    }

    virtio_scsi_complete_cmd_req(req);

}
