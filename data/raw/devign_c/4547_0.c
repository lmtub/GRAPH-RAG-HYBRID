static void scsi_do_read(SCSIDiskReq *r, int ret)

{

    SCSIDiskState *s = DO_UPCAST(SCSIDiskState, qdev, r->req.dev);

    SCSIDiskClass *sdc = (SCSIDiskClass *) object_get_class(OBJECT(s));



    assert (r->req.aiocb == NULL);



    if (r->req.io_canceled) {

        scsi_req_cancel_complete(&r->req);

        goto done;

    }



    if (ret < 0) {

        if (scsi_handle_rw_error(r, -ret, false)) {

            goto done;

        }

    }



    /* The request is used as the AIO opaque value, so add a ref.  */

    scsi_req_ref(&r->req);



    if (r->req.sg) {

        dma_acct_start(s->qdev.conf.blk, &r->acct, r->req.sg, BLOCK_ACCT_READ);

        r->req.resid -= r->req.sg->size;

        r->req.aiocb = dma_blk_io(blk_get_aio_context(s->qdev.conf.blk),

                                  r->req.sg, r->sector << BDRV_SECTOR_BITS,

                                  sdc->dma_readv, r, scsi_dma_complete, r,

                                  DMA_DIRECTION_FROM_DEVICE);

    } else {

        scsi_init_iovec(r, SCSI_DMA_BUF_SIZE);

        block_acct_start(blk_get_stats(s->qdev.conf.blk), &r->acct,

                         r->qiov.size, BLOCK_ACCT_READ);

        r->req.aiocb = sdc->dma_readv(r->sector, &r->qiov,

                                      scsi_read_complete, r, r);

    }



done:

    scsi_req_unref(&r->req);

}
