void virtio_scsi_handle_cmd_req_submit(VirtIOSCSI *s, VirtIOSCSIReq *req)

{

    if (scsi_req_enqueue(req->sreq)) {

        scsi_req_continue(req->sreq);

    }

    bdrv_io_unplug(req->sreq->dev->conf.bs);

    scsi_req_unref(req->sreq);

}
