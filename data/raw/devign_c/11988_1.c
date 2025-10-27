void scsi_req_cancel(SCSIRequest *req)

{

    trace_scsi_req_cancel(req->dev->id, req->lun, req->tag);

    if (!req->enqueued) {

        return;

    }



    scsi_req_ref(req);

    scsi_req_dequeue(req);

    req->io_canceled = true;

    if (req->aiocb) {

        blk_aio_cancel(req->aiocb);

    } else {

        scsi_req_cancel_complete(req);

    }

}