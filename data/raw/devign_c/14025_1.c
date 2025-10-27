static void scsi_disk_purge_requests(SCSIDiskState *s)

{

    SCSIDiskReq *r;



    while (!QTAILQ_EMPTY(&s->qdev.requests)) {

        r = DO_UPCAST(SCSIDiskReq, req, QTAILQ_FIRST(&s->qdev.requests));

        if (r->req.aiocb) {

            bdrv_aio_cancel(r->req.aiocb);

        }

        scsi_remove_request(r);

    }

}
