SCSIRequest *scsi_req_find(SCSIDevice *d, uint32_t tag)

{

    SCSIRequest *req;



    QTAILQ_FOREACH(req, &d->requests, next) {

        if (req->tag == tag) {

            return req;

        }

    }

    return NULL;

}
