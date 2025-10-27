static void scsi_write_same_complete(void *opaque, int ret)

{

    WriteSameCBData *data = opaque;

    SCSIDiskReq *r = data->r;

    SCSIDiskState *s = DO_UPCAST(SCSIDiskState, qdev, r->req.dev);



    assert(r->req.aiocb != NULL);

    r->req.aiocb = NULL;

    aio_context_acquire(blk_get_aio_context(s->qdev.conf.blk));

    if (scsi_disk_req_check_error(r, ret, true)) {

        goto done;

    }



    block_acct_done(blk_get_stats(s->qdev.conf.blk), &r->acct);



    data->nb_sectors -= data->iov.iov_len / 512;

    data->sector += data->iov.iov_len / 512;

    data->iov.iov_len = MIN(data->nb_sectors * 512, data->iov.iov_len);

    if (data->iov.iov_len) {

        block_acct_start(blk_get_stats(s->qdev.conf.blk), &r->acct,

                         data->iov.iov_len, BLOCK_ACCT_WRITE);

        /* Reinitialize qiov, to handle unaligned WRITE SAME request

         * where final qiov may need smaller size */

        qemu_iovec_init_external(&data->qiov, &data->iov, 1);

        r->req.aiocb = blk_aio_pwritev(s->qdev.conf.blk,

                                       data->sector << BDRV_SECTOR_BITS,

                                       &data->qiov, 0,

                                       scsi_write_same_complete, data);


        return;

    }



    scsi_req_complete(&r->req, GOOD);



done:

    scsi_req_unref(&r->req);

    qemu_vfree(data->iov.iov_base);

    g_free(data);


}