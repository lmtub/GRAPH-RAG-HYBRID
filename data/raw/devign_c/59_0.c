void virtio_scsi_handle_cmd_req_submit(VirtIOSCSI *s, VirtIOSCSIReq *req)

{

    SCSIRequest *sreq = req->sreq;

    if (scsi_req_enqueue(sreq)) {

        scsi_req_continue(sreq);

    }

    bdrv_io_unplug(sreq->dev->conf.bs);

    scsi_req_unref(sreq);

}
