static void scsi_write_data(SCSIRequest *req)

{

    SCSIDiskReq *r = DO_UPCAST(SCSIDiskReq, req, req);

    SCSIDiskState *s = DO_UPCAST(SCSIDiskState, qdev, r->req.dev);

    uint32_t n;



    /* No data transfer may already be in progress */

    assert(r->req.aiocb == NULL);



    /* The request is used as the AIO opaque value, so add a ref.  */

    scsi_req_ref(&r->req);

    if (r->req.cmd.mode != SCSI_XFER_TO_DEV) {

        DPRINTF("Data transfer direction invalid\n");

        scsi_write_complete(r, -EINVAL);

        return;

    }



    if (!r->req.sg && !r->qiov.size) {

        /* Called for the first time.  Ask the driver to send us more data.  */

        r->started = true;

        scsi_write_complete(r, 0);

        return;

    }

    if (s->tray_open) {

        scsi_write_complete(r, -ENOMEDIUM);

        return;

    }



    if (r->req.cmd.buf[0] == VERIFY_10 || r->req.cmd.buf[0] == VERIFY_12 ||

        r->req.cmd.buf[0] == VERIFY_16) {

        if (r->req.sg) {

            scsi_dma_complete(r, 0);

        } else {

            scsi_write_complete(r, 0);

        }

        return;

    }



    if (r->req.sg) {

        dma_acct_start(s->qdev.conf.bs, &r->acct, r->req.sg, BDRV_ACCT_WRITE);

        r->req.resid -= r->req.sg->size;

        r->req.aiocb = dma_bdrv_write(s->qdev.conf.bs, r->req.sg, r->sector,

                                      scsi_dma_complete, r);

    } else {

        n = r->qiov.size / 512;

        bdrv_acct_start(s->qdev.conf.bs, &r->acct, n * BDRV_SECTOR_SIZE, BDRV_ACCT_WRITE);

        r->req.aiocb = bdrv_aio_writev(s->qdev.conf.bs, r->sector, &r->qiov, n,

                                       scsi_write_complete, r);

    }

}
