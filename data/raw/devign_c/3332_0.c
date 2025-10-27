static void scsi_write_complete(void * opaque, int ret)

{

    SCSIDiskReq *r = (SCSIDiskReq *)opaque;

    SCSIDiskState *s = DO_UPCAST(SCSIDiskState, qdev, r->req.dev);

    uint32_t n;



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



    n = r->qiov.size / 512;

    r->sector += n;

    r->sector_count -= n;

    if (r->sector_count == 0) {

        scsi_write_do_fua(r);

        return;

    } else {

        scsi_init_iovec(r, SCSI_DMA_BUF_SIZE);

        DPRINTF("Write complete tag=0x%x more=%zd\n", r->req.tag, r->qiov.size);

        scsi_req_data(&r->req, r->qiov.size);

    }



done:

    if (!r->req.io_canceled) {

        scsi_req_unref(&r->req);

    }

}
