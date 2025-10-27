static void scsi_dma_complete_noio(void *opaque, int ret)

{

    SCSIDiskReq *r = (SCSIDiskReq *)opaque;

    SCSIDiskState *s = DO_UPCAST(SCSIDiskState, qdev, r->req.dev);



    if (r->req.aiocb != NULL) {

        r->req.aiocb = NULL;

        block_acct_done(bdrv_get_stats(s->qdev.conf.bs), &r->acct);

    }

    if (r->req.io_canceled) {

        goto done;

    }



    if (ret < 0) {

        if (scsi_handle_rw_error(r, -ret)) {

            goto done;

        }

    }



    r->sector += r->sector_count;

    r->sector_count = 0;

    if (r->req.cmd.mode == SCSI_XFER_TO_DEV) {

        scsi_write_do_fua(r);

        return;

    } else {

        scsi_req_complete(&r->req, GOOD);

    }



done:

    if (!r->req.io_canceled) {

        scsi_req_unref(&r->req);

    }

}
