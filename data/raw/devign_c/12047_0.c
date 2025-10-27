static void scsi_do_read(void *opaque, int ret)

{

    SCSIDiskReq *r = opaque;

    SCSIDiskState *s = DO_UPCAST(SCSIDiskState, qdev, r->req.dev);

    uint32_t n;



    if (r->req.aiocb != NULL) {

        r->req.aiocb = NULL;

        bdrv_acct_done(s->qdev.conf.bs, &r->acct);

    }



    if (ret < 0) {

        if (scsi_handle_rw_error(r, -ret)) {

            goto done;

        }

    }



    if (r->req.io_canceled) {

        return;

    }



    /* The request is used as the AIO opaque value, so add a ref.  */

    scsi_req_ref(&r->req);



    if (r->req.sg) {

        dma_acct_start(s->qdev.conf.bs, &r->acct, r->req.sg, BDRV_ACCT_READ);

        r->req.resid -= r->req.sg->size;

        r->req.aiocb = dma_bdrv_read(s->qdev.conf.bs, r->req.sg, r->sector,

                                     scsi_dma_complete, r);

    } else {

        n = scsi_init_iovec(r, SCSI_DMA_BUF_SIZE);

        bdrv_acct_start(s->qdev.conf.bs, &r->acct, n * BDRV_SECTOR_SIZE, BDRV_ACCT_READ);

        r->req.aiocb = bdrv_aio_readv(s->qdev.conf.bs, r->sector, &r->qiov, n,

                                      scsi_read_complete, r);

    }



done:

    if (!r->req.io_canceled) {

        scsi_req_unref(&r->req);

    }

}
