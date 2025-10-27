void scsi_req_data(SCSIRequest *req, int len)

{

    trace_scsi_req_data(req->dev->id, req->lun, req->tag, len);

    req->bus->ops->complete(req->bus, SCSI_REASON_DATA, req->tag, len);

}
