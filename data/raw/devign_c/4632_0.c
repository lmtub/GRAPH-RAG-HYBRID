static void virtio_blk_handle_write(VirtIOBlockReq *req, MultiReqBuffer *mrb)

{

    BlockRequest *blkreq;

    uint64_t sector;



    sector = virtio_ldq_p(VIRTIO_DEVICE(req->dev), &req->out.sector);



    trace_virtio_blk_handle_write(req, sector, req->qiov.size / 512);



    if (!virtio_blk_sect_range_ok(req->dev, sector, req->qiov.size)) {

        virtio_blk_req_complete(req, VIRTIO_BLK_S_IOERR);

        virtio_blk_free_request(req);

        return;

    }



    block_acct_start(bdrv_get_stats(req->dev->bs), &req->acct, req->qiov.size,

                     BLOCK_ACCT_WRITE);



    if (mrb->num_writes == 32) {

        virtio_submit_multiwrite(req->dev->bs, mrb);

    }



    blkreq = &mrb->blkreq[mrb->num_writes];

    blkreq->sector = sector;

    blkreq->nb_sectors = req->qiov.size / BDRV_SECTOR_SIZE;

    blkreq->qiov = &req->qiov;

    blkreq->cb = virtio_blk_rw_complete;

    blkreq->opaque = req;

    blkreq->error = 0;



    mrb->num_writes++;

}
