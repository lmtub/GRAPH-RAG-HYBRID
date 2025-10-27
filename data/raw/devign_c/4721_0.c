static void scsi_read_data(SCSIRequest *req)

{

    SCSIDiskReq *r = DO_UPCAST(SCSIDiskReq, req, req);

    SCSIDiskState *s = DO_UPCAST(SCSIDiskState, qdev, r->req.dev);

    bool first;



    DPRINTF("Read sector_count=%d\n", r->sector_count);

    if (r->sector_count == 0) {

        /* This also clears the sense buffer for REQUEST SENSE.  */

        scsi_req_complete(&r->req, GOOD);

        return;

    }



    /* No data transfer may already be in progress */

    assert(r->req.aiocb == NULL);



    /* The request is used as the AIO opaque value, so add a ref.  */

    scsi_req_ref(&r->req);

    if (r->req.cmd.mode == SCSI_XFER_TO_DEV) {

        DPRINTF("Data transfer direction invalid\n");

        scsi_read_complete(r, -EINVAL);

        return;

    }



    if (s->tray_open) {

        scsi_read_complete(r, -ENOMEDIUM);

        return;

    }



    first = !r->started;

    r->started = true;

    if (first && r->need_fua_emulation) {

        block_acct_start(blk_get_stats(s->qdev.conf.blk), &r->acct, 0,

                         BLOCK_ACCT_FLUSH);

        r->req.aiocb = blk_aio_flush(s->qdev.conf.blk, scsi_do_read_cb, r);

    } else {

        scsi_do_read(r, 0);

    }

}
