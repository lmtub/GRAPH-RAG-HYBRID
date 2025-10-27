static void scsi_read_request(SCSIDiskReq *r)

{

    SCSIDiskState *s = DO_UPCAST(SCSIDiskState, qdev, r->req.dev);

    uint32_t n;



    if (r->sector_count == (uint32_t)-1) {

        DPRINTF("Read buf_len=%zd\n", r->iov.iov_len);

        r->sector_count = 0;

        scsi_req_data(&r->req, r->iov.iov_len);

        return;

    }

    DPRINTF("Read sector_count=%d\n", r->sector_count);

    if (r->sector_count == 0) {

        scsi_command_complete(r, GOOD, NO_SENSE);

        return;

    }



    /* No data transfer may already be in progress */

    assert(r->req.aiocb == NULL);



    n = r->sector_count;

    if (n > SCSI_DMA_BUF_SIZE / 512)

        n = SCSI_DMA_BUF_SIZE / 512;



    r->iov.iov_len = n * 512;

    qemu_iovec_init_external(&r->qiov, &r->iov, 1);

    r->req.aiocb = bdrv_aio_readv(s->bs, r->sector, &r->qiov, n,

                              scsi_read_complete, r);

    if (r->req.aiocb == NULL) {

        scsi_read_complete(r, -EIO);

    }

}
