void scsi_req_complete(SCSIRequest *req)

{

    assert(req->status != -1);

    scsi_req_ref(req);

    scsi_req_dequeue(req);

    req->bus->ops->complete(req->bus, SCSI_REASON_DONE,

                            req->tag,

                            req->status);

    scsi_req_unref(req);

}
