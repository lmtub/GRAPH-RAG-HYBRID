void scsi_req_cancel_async(SCSIRequest *req, Notifier *notifier)

{

    trace_scsi_req_cancel(req->dev->id, req->lun, req->tag);

    if (notifier) {

        notifier_list_add(&req->cancel_notifiers, notifier);











    scsi_req_ref(req);

    scsi_req_dequeue(req);

    req->io_canceled = true;

    if (req->aiocb) {

        blk_aio_cancel_async(req->aiocb);

    } else {

        scsi_req_cancel_complete(req);

