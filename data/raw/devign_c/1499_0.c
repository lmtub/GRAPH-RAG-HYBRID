int32_t scsi_req_enqueue(SCSIRequest *req, uint8_t *buf)

{

    int32_t rc;



    assert(!req->enqueued);

    scsi_req_ref(req);

    req->enqueued = true;

    QTAILQ_INSERT_TAIL(&req->dev->requests, req, next);



    scsi_req_ref(req);

    rc = req->ops->send_command(req, buf);

    scsi_req_unref(req);

    return rc;

}
