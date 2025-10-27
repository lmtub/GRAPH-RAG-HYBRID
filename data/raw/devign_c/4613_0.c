static void virtio_scsi_fail_cmd_req(VirtIOSCSIReq *req)

{

    req->resp.cmd->response = VIRTIO_SCSI_S_FAILURE;

    virtio_scsi_complete_cmd_req(req);

}
