void scsi_req_dequeue(SCSIRequest *req)

{

    trace_scsi_req_dequeue(req->dev->id, req->lun, req->tag);

    if (req->enqueued) {

        QTAILQ_REMOVE(&req->dev->requests, req, next);

        req->enqueued = false;

        scsi_req_unref(req);

    }

}
