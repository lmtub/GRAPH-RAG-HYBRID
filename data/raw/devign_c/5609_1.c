SCSIRequest *scsi_req_alloc(size_t size, SCSIDevice *d, uint32_t tag, uint32_t lun)

{

    SCSIRequest *req;



    req = qemu_mallocz(size);

    /* Two references: one is passed back to the HBA, one is in d->requests.  */

    req->refcount = 2;

    req->bus = scsi_bus_from_device(d);

    req->dev = d;

    req->tag = tag;

    req->lun = lun;

    req->status = -1;

    req->enqueued = true;

    trace_scsi_req_alloc(req->dev->id, req->lun, req->tag);

    QTAILQ_INSERT_TAIL(&d->requests, req, next);

    return req;

}
