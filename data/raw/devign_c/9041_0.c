static void scsi_dma_complete(void *opaque, int ret)

{

    SCSIDiskReq *r = (SCSIDiskReq *)opaque;

    SCSIDiskState *s = DO_UPCAST(SCSIDiskState, qdev, r->req.dev);



    bdrv_acct_done(s->qdev.conf.bs, &r->acct);



    if (ret) {

        if (scsi_handle_rw_error(r, -ret)) {

            goto done;

        }

    }



    r->sector += r->sector_count;

    r->sector_count = 0;

    scsi_req_complete(&r->req, GOOD);



done:

    if (!r->req.io_canceled) {

        scsi_req_unref(&r->req);

    }

}
